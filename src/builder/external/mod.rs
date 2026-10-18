//! External (out-of-core) build support: spilling inserted keys to disk and building a set with
//! a most-significant-byte radix sort, so the full key set never has to fit in memory.
//!
//! This module is private to [`builder`](crate::builder); it is reached through
//! [`ByteStringSet::external_builder`](crate::ByteStringSet::external_builder).

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::path::PathBuf;

use memmap2::Mmap;
use tempfile::TempDir;

use super::{emit, sort_dedup};
use crate::{ByteStringSet, Error, HEADER_LEN, Storage, build_offsets, write_header};

/// Read a whole bucket of `N`-byte keys (`size` bytes) from `file` into memory.
fn read_keys<const N: usize>(file: File, size: u64) -> std::io::Result<Vec<[u8; N]>> {
    let count = usize::try_from(size / N as u64)
        .expect("bucket size is bounded by a usize max_bucket_bytes");
    let mut keys = Vec::new();
    super::read_keys_into(&mut BufReader::new(file), count, N, &mut keys)?;
    Ok(keys)
}

/// An open partition file accumulating the keys that share one leading byte value.
#[derive(Debug)]
struct Partition<const N: usize> {
    path: PathBuf,
    writer: BufWriter<File>,
    bytes: u64,
    /// The first key written to this partition.
    first: [u8; N],
    /// Length of the retained prefix shared by every key written to this partition so far.
    shared: usize,
}

/// State for one external build pass: most-significant-byte partitioning plus the
/// output writer the sorted, deduplicated prefixes stream to.
#[derive(Debug)]
struct ExternalBuild<const N: usize> {
    prefix_len: usize,
    /// Largest bucket, in bytes, that may be loaded and sorted in memory.
    max_bucket_bytes: u64,
    /// Directory holding intermediate partition files; removed when this pass is dropped.
    partition_dir: TempDir,
    /// Monotonic counter used to give partition files unique names.
    next_file_id: u64,
    /// Destination data file (header already written).
    out: BufWriter<File>,
}

/// A closed partition waiting to be processed; all its keys share their first `depth` bytes.
#[derive(Debug)]
struct Bucket {
    path: PathBuf,
    size: u64,
    depth: usize,
}

impl<const N: usize> ExternalBuild<N> {
    /// Process buckets in ascending prefix order without growing the call stack. Remove each
    /// input partition before visiting its children so ancestor copies do not accumulate on disk.
    fn run(&mut self, file: File, size: u64) -> Result<(), Error> {
        let mut pending = self.process(file, size, 0)?;
        while let Some(bucket) = pending.pop() {
            let children = self.process(File::open(&bucket.path)?, bucket.size, bucket.depth)?;
            fs::remove_file(bucket.path)?;
            pending.extend(children);
        }
        Ok(())
    }

    /// Emit a bucket that fits in memory or shares its entire prefix. Otherwise partition it,
    /// returning children in reverse order for the stack of pending buckets.
    fn process(&mut self, mut file: File, size: u64, depth: usize) -> Result<Vec<Bucket>, Error> {
        if depth == self.prefix_len && size != 0 {
            let mut key = [0u8; N];
            file.read_exact(&mut key)?;
            self.out.write_all(&key[..self.prefix_len])?;
            Ok(Vec::new())
        } else if size <= self.max_bucket_bytes {
            let mut keys = read_keys::<N>(file, size)?;
            sort_dedup(&mut keys, self.prefix_len);
            emit(&keys, self.prefix_len, &mut self.out)?;
            Ok(Vec::new())
        } else {
            self.partition(file, size, depth)
        }
    }

    /// Split `file` (`size` bytes) into per-byte-value partition files keyed on byte `depth`,
    /// returning non-empty partitions in descending byte order.
    ///
    /// Each child's depth is the length of the prefix its keys share, so a child that is
    /// partitioned again always splits on its first byte.
    fn partition(&mut self, file: File, size: u64, depth: usize) -> Result<Vec<Bucket>, Error> {
        // One slot per possible byte value; partition files are created lazily so skewed data
        // does not allocate 256 files per level.
        let mut buckets: Vec<Option<Partition<N>>> = (0..256).map(|_| None).collect();
        let mut reader = BufReader::new(file);
        let mut key = [0u8; N];
        for _ in 0..size / N as u64 {
            reader.read_exact(&mut key)?;
            let slot = &mut buckets[key[depth] as usize];
            let bucket = if let Some(bucket) = slot {
                // Bytes before `depth` are shared by the whole input, and byte `depth` by the
                // partition, so only the bytes after it need comparing.
                bucket.shared = depth
                    + 1
                    + key[depth + 1..bucket.shared]
                        .iter()
                        .zip(&bucket.first[depth + 1..bucket.shared])
                        .take_while(|(left, right)| left == right)
                        .count();
                bucket
            } else {
                let path = self
                    .partition_dir
                    .path()
                    .join(self.next_file_id.to_string());
                self.next_file_id += 1;
                slot.insert(Partition {
                    writer: BufWriter::new(File::create(&path)?),
                    path,
                    bytes: 0,
                    first: key,
                    shared: self.prefix_len,
                })
            };
            bucket.writer.write_all(&key)?;
            bucket.bytes += N as u64;
        }
        let mut children = Vec::with_capacity(buckets.iter().filter(|slot| slot.is_some()).count());
        for partition in buckets.into_iter().rev().flatten() {
            // `into_inner` flushes the buffer and hands back the `File`, which closes on drop.
            partition
                .writer
                .into_inner()
                .map_err(std::io::IntoInnerError::into_error)?;
            children.push(Bucket {
                path: partition.path,
                size: partition.bytes,
                depth: partition.shared,
            });
        }
        Ok(children)
    }
}

/// Insert-time state of an external-sort builder: keys stream to an anonymous spill file.
#[derive(Debug)]
pub(super) struct SpillState<const N: usize> {
    // Absent after an append fails, since a partial key may have reached the file.
    writer: Option<BufWriter<File>>,
    max_bucket_bytes: usize,
}

impl<const N: usize> SpillState<N> {
    /// Create a spill file that inserted keys stream to.
    ///
    /// # Arguments
    ///
    /// * `max_bucket_bytes` - Largest bucket of raw keys, in bytes, that [`build`] may load and
    ///   sort in memory at once; values smaller than `N` are treated as `N`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the spill file cannot be created.
    ///
    /// [`build`]: SpillState::build
    pub(super) fn new(max_bucket_bytes: usize) -> Result<Self, Error> {
        Ok(Self {
            writer: Some(BufWriter::new(tempfile::tempfile()?)),
            max_bucket_bytes: max_bucket_bytes.max(N),
        })
    }

    /// Append a key to the spill file.
    ///
    /// # Errors
    ///
    /// Returns an error if writing to the spill file fails.
    pub(super) fn insert(&mut self, key: &[u8; N]) -> std::io::Result<()> {
        self.append(|writer| writer.write_all(key))
    }

    /// Append entries from a validated header, padding short prefixes to full keys.
    ///
    /// # Errors
    ///
    /// Returns an error if reading or writing fails.
    pub(super) fn import(
        &mut self,
        reader: &mut impl Read,
        count: u64,
        prefix_len: usize,
    ) -> std::io::Result<()> {
        self.append(|writer| {
            if prefix_len == N {
                let bytes = count * N as u64;
                let copied = std::io::copy(&mut reader.take(bytes), writer)?;
                if copied == bytes {
                    Ok(())
                } else {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "imported data file was truncated",
                    ))
                }
            } else {
                let mut key = [0u8; N];
                for _ in 0..count {
                    reader.read_exact(&mut key[..prefix_len])?;
                    writer.write_all(&key)?;
                }
                Ok(())
            }
        })
    }

    /// Restore the writer only after a complete append, so failed operations cannot leave
    /// partial keys available to later inserts or builds.
    fn append(
        &mut self,
        write: impl FnOnce(&mut BufWriter<File>) -> std::io::Result<()>,
    ) -> std::io::Result<()> {
        let mut writer = self.writer.take().ok_or_else(Self::failed)?;
        write(&mut writer)?;
        self.writer = Some(writer);
        Ok(())
    }

    fn failed() -> std::io::Error {
        std::io::Error::other("external builder cannot be used after a failed insert or import")
    }

    /// Finish an external build: externally sort the spill file and memory-map the result.
    ///
    /// # Arguments
    ///
    /// * `prefix_len` - How many leading bytes of each key to keep; already validated by the
    ///   caller to be within `3..=N`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TooManyEntries`] if the deduplicated entry count overflows the `u32`
    /// offset table, and [`Error::Io`] if a temporary file cannot be read or written.
    #[expect(unsafe_code, reason = "mapping a private temporary file")]
    pub(super) fn build(self, prefix_len: usize) -> Result<ByteStringSet<N>, Error> {
        let mut spill_file = self
            .writer
            .ok_or_else(Self::failed)?
            .into_inner()
            .map_err(std::io::IntoInnerError::into_error)?;
        let spilled = spill_file.metadata()?.len();
        // The spill file's cursor sits at the end after writing; rewind before reading it back.
        spill_file.rewind()?;
        let mut pass = ExternalBuild::<N> {
            prefix_len,
            max_bucket_bytes: self.max_bucket_bytes as u64,
            partition_dir: tempfile::tempdir()?,
            next_file_id: 0,
            out: BufWriter::new(tempfile::tempfile()?),
        };
        write_header(&mut pass.out, prefix_len)?;
        pass.run(spill_file, spilled)?;
        let out_file = pass
            .out
            .into_inner()
            .map_err(std::io::IntoInnerError::into_error)?;
        // SAFETY: this process exclusively owns the temporary file and will not modify it again.
        // tempfile unlinks it on Unix and disables sharing on Windows; the map keeps it alive.
        let map = unsafe { Mmap::map(&out_file)? };
        // Rebuilding the offsets from the mapped data also re-validates that the emission is
        // sorted and deduplicated, exactly as `ByteStringSet::read` does for files loaded from
        // disk.
        let offsets = build_offsets(&map[HEADER_LEN..], prefix_len)?;
        Ok(ByteStringSet {
            prefix_len,
            data: Storage::Mapped(map),
            offsets,
        })
    }
}

#[cfg(test)]
mod tests;
