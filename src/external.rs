//! External (out-of-core) build support: spilling inserted keys to disk and building a set with
//! a recursive most-significant-byte radix sort, so the full key set never has to fit in memory.
//!
//! This module is crate-private; it is reached through
//! [`ByteStringSet::external_builder`](crate::ByteStringSet::external_builder).

use memmap2::Mmap;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Seek, Write};
use std::path::PathBuf;
use tempfile::TempDir;

use crate::{
    ByteStringSet, Error, HEADER_LEN, Storage, build_offsets, sort_dedup_emit, write_header,
};

/// Read a whole bucket of `N`-byte keys (`size` bytes) from `file` into memory.
fn read_keys<const N: usize>(file: File, size: u64) -> io::Result<Vec<[u8; N]>> {
    let count = usize::try_from(size / N as u64)
        .expect("bucket size is bounded by a usize max_bucket_bytes");
    let mut reader = BufReader::new(file);
    let mut keys = Vec::with_capacity(count);
    let mut key = [0u8; N];
    for _ in 0..count {
        reader.read_exact(&mut key)?;
        // `[u8; N]` is `Copy`, so pushing copies the array rather than moving it.
        keys.push(key);
    }
    Ok(keys)
}

/// An open partition file accumulating the keys that share one leading byte value.
#[derive(Debug)]
struct Partition {
    path: PathBuf,
    writer: BufWriter<File>,
    bytes: u64,
}

/// State for one external build pass: recursive most-significant-byte partitioning plus the
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

impl<const N: usize> ExternalBuild<N> {
    /// Emit the sorted, deduplicated prefixes of `file`, whose `size` bytes hold keys that all
    /// share their first `depth` bytes.
    ///
    /// Buckets no larger than `max_bucket_bytes` are loaded and sorted in memory; larger ones
    /// are partitioned on byte `depth` and recursed into in ascending byte order, which keeps
    /// the overall emission sorted.
    fn run(&mut self, mut file: File, size: u64, depth: usize) -> Result<(), Error> {
        if size <= self.max_bucket_bytes {
            let mut keys = read_keys::<N>(file, size)?;
            sort_dedup_emit(&mut keys, self.prefix_len, &mut self.out)?;
        } else if depth == self.prefix_len {
            // Every key in this bucket shares its whole `P`-byte prefix, so the bucket
            // contributes exactly one entry no matter how large it is; read one key instead of
            // partitioning further.
            let mut key = [0u8; N];
            file.read_exact(&mut key)?;
            self.out.write_all(&key[..self.prefix_len])?;
        } else {
            for (path, child_size) in self.partition(file, size, depth)? {
                let child = File::open(&path)?;
                self.run(child, child_size, depth + 1)?;
                // Reclaim disk as each partition is consumed instead of waiting for the
                // `TempDir` to be dropped.
                fs::remove_file(&path)?;
            }
        }
        Ok(())
    }

    /// Split `file` (`size` bytes) into per-byte-value partition files keyed on byte `depth`,
    /// returning the non-empty partitions as `(path, size)` pairs in ascending byte order.
    fn partition(
        &mut self,
        file: File,
        size: u64,
        depth: usize,
    ) -> Result<Vec<(PathBuf, u64)>, Error> {
        // One slot per possible byte value; partition files are created lazily so skewed data
        // does not allocate 256 files per level.
        let mut buckets: Vec<Option<Partition>> = (0..256).map(|_| None).collect();
        let mut reader = BufReader::new(file);
        let mut key = [0u8; N];
        for _ in 0..size / N as u64 {
            reader.read_exact(&mut key)?;
            let slot = &mut buckets[key[depth] as usize];
            let bucket = if let Some(bucket) = slot {
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
                })
            };
            bucket.writer.write_all(&key)?;
            bucket.bytes += N as u64;
        }
        let mut children = Vec::with_capacity(buckets.iter().filter(|slot| slot.is_some()).count());
        for partition in buckets.into_iter().flatten() {
            // `into_inner` flushes the buffer and hands back the `File`, which closes on drop.
            partition
                .writer
                .into_inner()
                .map_err(io::IntoInnerError::into_error)?;
            children.push((partition.path, partition.bytes));
        }
        Ok(children)
    }
}

/// Insert-time state of an external-sort builder: keys stream to an anonymous spill file.
#[derive(Debug)]
pub struct SpillState {
    writer: BufWriter<File>,
    max_bucket_bytes: usize,
}

impl SpillState {
    /// Create a spill file that inserted keys stream to.
    ///
    /// # Arguments
    ///
    /// * `max_bucket_bytes` - Largest bucket of raw keys, in bytes, that [`build`] may load and
    ///   sort in memory at once; the caller has already clamped it to at least one key.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the spill file cannot be created.
    ///
    /// [`build`]: SpillState::build
    pub(crate) fn new(max_bucket_bytes: usize) -> Result<Self, Error> {
        Ok(Self {
            writer: BufWriter::new(tempfile::tempfile()?),
            max_bucket_bytes,
        })
    }

    /// Append a raw `N`-byte key to the spill file.
    ///
    /// # Errors
    ///
    /// Returns an error if writing to the spill file fails.
    pub(crate) fn insert(&mut self, key: &[u8]) -> io::Result<()> {
        self.writer.write_all(key)
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
    pub(crate) fn build<const N: usize>(
        self,
        prefix_len: usize,
    ) -> Result<ByteStringSet<N>, Error> {
        let mut spill_file = self
            .writer
            .into_inner()
            .map_err(io::IntoInnerError::into_error)?;
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
        pass.run(spill_file, spilled, 0)?;
        let out_file = pass
            .out
            .into_inner()
            .map_err(io::IntoInnerError::into_error)?;
        // SAFETY: the anonymous temporary file was created, written, and is owned exclusively by
        // this process; it is already unlinked, so no other process can open or modify it while
        // mapped.
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
