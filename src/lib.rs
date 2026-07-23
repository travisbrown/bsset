#![warn(clippy::all, clippy::pedantic, clippy::nursery, rust_2018_idioms)]
//! A byte-string set for fixed-length keys with prefix-based membership.
//!
//! Keys have a type-level length `N`. Each set also has a runtime prefix length (called `P` here).
//! When `P == N` membership checks are exact.
//!
//! # Layout
//!
//! The set is two flat buffers:
//!
//! * a byte buffer of the sorted, deduplicated `P`-byte prefixes laid out back to back (in
//!   `P`-byte strides), and
//! * an offset table of `2^24 + 1` `u32` entry indices, keyed by the first 3 bytes of a prefix,
//!   delimiting each bucket of the byte buffer.
//!
//! A lookup is one table read plus a binary search inside a single bucket. The table costs a
//! fixed 64 MiB per set, which pays off for large sets.
//!
//! Building normally buffers all keys in memory. For sets that do not fit,
//! [`ByteStringSetBuilder::with_max_bucket_bytes`] spills keys to temporary files and builds the
//! set with an external most-significant-byte radix sort instead.
//!
//! # Examples
//!
//! ```
//! use bsset::ByteStringSet;
//!
//! let mut builder = ByteStringSet::<4>::builder();
//! builder.insert(*b"abcd")?;
//! builder.insert(*b"wxyz")?;
//!
//! // Prefix length 4 == key length 4, so lookups are exact.
//! let set = builder.build(4)?;
//! assert!(set.lookup(*b"abcd"));
//! assert!(!set.lookup(*b"aaaa"));
//! # Ok::<(), bsset::Error>(())
//! ```

use memmap2::Mmap;
use std::cmp::Ordering;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use thiserror::Error;

mod external;

/// Number of bytes consumed by the first-level offset table.
const OFFSET_TABLE_PREFIX_LEN: usize = 3;

/// One slot per possible 3-byte prefix, plus a trailing sentinel so every bucket `i` is delimited
/// by `offsets[i]..offsets[i + 1]`.
const OFFSET_TABLE_LEN: usize = (1 << (8 * OFFSET_TABLE_PREFIX_LEN)) + 1;

/// Magic bytes identifying a bsset data file.
const MAGIC: [u8; 4] = *b"BSST";

/// Current data file format version.
const FORMAT_VERSION: u8 = 1;

/// Data file header: bytes `0..4` magic, `4` version, `5..8` reserved (zero), `8..16` prefix
/// length as little-endian `u64`.
const HEADER_LEN: usize = 16;

/// Errors produced when building or loading a [`ByteStringSet`].
#[derive(Debug, Error)]
pub enum Error {
    /// The requested prefix length is outside `3..=N`.
    #[error("prefix length {prefix_len} out of range; must be within 3..={key_len}")]
    InvalidPrefixLen {
        /// The rejected prefix length.
        prefix_len: usize,
        /// The type-level key length `N`.
        key_len: usize,
    },
    /// A data file's size is not a whole number of `P`-byte strides.
    #[error("data length {data_len} is not a multiple of prefix length {prefix_len}")]
    Misaligned {
        /// Total data size in bytes.
        data_len: usize,
        /// The stride the data was expected to be laid out in.
        prefix_len: usize,
    },
    /// A data file's entries are not strictly ascending (sorted and deduped).
    #[error("data is not sorted and deduplicated at entry {index}")]
    Unsorted {
        /// Index of the first entry that is `<=` its predecessor.
        index: usize,
    },
    /// The entry count does not fit the `u32` offset table.
    #[error("entry count {count} exceeds the u32 offset table range")]
    TooManyEntries {
        /// Number of entries encountered.
        count: usize,
    },
    /// A data file is missing the expected header magic or is truncated.
    #[error("not a bsset data file (missing or truncated header)")]
    BadHeader,
    /// A data file uses a format version this crate does not understand.
    #[error("unsupported data file format version {version}")]
    UnsupportedVersion {
        /// The version found in the file header.
        version: u8,
    },
    /// An underlying filesystem operation failed.
    #[error(transparent)]
    Io(#[from] io::Error),
}

/// Backing storage for the prefix buffer: built in memory or mapped from disk.
#[derive(Debug)]
enum Storage {
    Owned(Vec<u8>),
    Mapped(Mmap),
}

impl Storage {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            // Mapped files start with the header; `read` has already checked the map is at least
            // `HEADER_LEN` bytes long.
            Self::Mapped(map) => &map[HEADER_LEN..],
        }
    }
}

/// Offset table slot for a prefix: its first 3 bytes as a `usize`.
fn table_index(prefix: &[u8]) -> usize {
    u32::from_be_bytes([0, prefix[0], prefix[1], prefix[2]]) as usize
}

/// Build the 3-byte-prefix offset table for sorted `prefix_len`-stride data.
///
/// Validates that entries are strictly ascending (i.e. sorted and deduplicated), which the
/// table's correctness depends on. `data.len()` must already be a multiple of `prefix_len`.
///
/// # Errors
///
/// Returns [`Error::TooManyEntries`] if the entry count overflows `u32`, and [`Error::Unsorted`]
/// if any entry is not strictly greater than its predecessor.
fn build_offsets(data: &[u8], prefix_len: usize) -> Result<Vec<u32>, Error> {
    let count = data.len() / prefix_len;

    if count > u32::MAX as usize {
        Err(Error::TooManyEntries { count })
    } else {
        let mut offsets = vec![0u32; OFFSET_TABLE_LEN];
        let mut previous: Option<&[u8]> = None;
        for (index, entry) in data.chunks_exact(prefix_len).enumerate() {
            if previous.is_some_and(|previous| previous >= entry) {
                return Err(Error::Unsorted { index });
            }
            offsets[table_index(entry) + 1] += 1;
            previous = Some(entry);
        }
        // Convert per-bucket counts into cumulative start offsets, so bucket `i` spans entry
        // indices `offsets[i]..offsets[i + 1]`.
        let mut running = 0u32;
        for slot in &mut offsets {
            running += *slot;
            *slot = running;
        }
        Ok(offsets)
    }
}

/// Write the data file header (magic, format version, and prefix length) to `out`.
fn write_header(out: &mut impl Write, prefix_len: usize) -> io::Result<()> {
    let mut header = [0u8; HEADER_LEN];
    header[..4].copy_from_slice(&MAGIC);
    header[4] = FORMAT_VERSION;
    header[8..16].copy_from_slice(&(prefix_len as u64).to_le_bytes());
    out.write_all(&header)
}

/// Sort `keys`, drop all but the first key of each distinct `prefix_len`-byte prefix, and write
/// the surviving prefixes to `out` in order.
fn sort_dedup_emit<const N: usize>(
    keys: &mut Vec<[u8; N]>,
    prefix_len: usize,
    out: &mut impl Write,
) -> io::Result<()> {
    // Lexicographic sort of the full keys also sorts their prefixes, so keys sharing a prefix
    // end up adjacent for `dedup_by` below.
    keys.sort_unstable();
    // `dedup_by` drops an element when the closure says it matches the element kept immediately
    // before it — here, same first `prefix_len` bytes.
    keys.dedup_by(|next, kept| next[..prefix_len] == kept[..prefix_len]);
    for key in keys.iter() {
        out.write_all(&key[..prefix_len])?;
    }
    Ok(())
}

/// Accumulates `[u8; N]` keys and builds a [`ByteStringSet`].
///
/// Obtained from [`ByteStringSet::builder`] (in-memory) or
/// [`ByteStringSetBuilder::with_max_bucket_bytes`] (external sort). Duplicate insertions are
/// fine; they are removed during [`build`].
///
/// [`build`]: ByteStringSetBuilder::build
#[derive(Debug, Default)]
pub struct ByteStringSetBuilder<const N: usize> {
    /// Keys buffered in memory; unused (always empty) when `spill` is active.
    keys: Vec<[u8; N]>,
    /// External-sort state; `None` for the in-memory builder.
    spill: Option<external::SpillState>,
}

impl<const N: usize> ByteStringSetBuilder<N> {
    /// Create a builder that uses an external most-significant-byte radix sort so the full key
    /// set never has to fit in memory.
    ///
    /// Inserted keys stream to an anonymous spill file instead of a `Vec`. [`build`] then
    /// recursively partitions the spilled keys by successive leading bytes until each bucket is
    /// at most `max_bucket_bytes`, sorts one bucket at a time in memory, and streams the
    /// deduplicated prefixes to a temporary data file that backs the finished set as a memory
    /// map. Peak memory during the build is therefore roughly `max_bucket_bytes` plus the fixed
    /// offset table.
    ///
    /// Temporary files are created in the system temporary directory (honoring the `TMPDIR`
    /// environment variable); if that is a RAM-backed filesystem such as a `tmpfs` `/tmp`, point
    /// `TMPDIR` at a disk-backed directory to get the full benefit. Values of `max_bucket_bytes`
    /// smaller than `N` are treated as `N`, since a bucket must hold at least one key.
    ///
    /// # Arguments
    ///
    /// * `max_bucket_bytes` - Largest bucket of raw `N`-byte keys, in bytes, that [`build`] may
    ///   load and sort in memory at once.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the spill file cannot be created.
    ///
    /// [`build`]: ByteStringSetBuilder::build
    pub fn with_max_bucket_bytes(max_bucket_bytes: usize) -> Result<Self, Error> {
        Ok(Self {
            keys: Vec::new(),
            spill: Some(external::SpillState::new(max_bucket_bytes.max(N))?),
        })
    }

    /// Add a key to the set under construction.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to insert.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if writing to the spill file fails; in-memory builders (from
    /// [`ByteStringSet::builder`]) never fail.
    pub fn insert(&mut self, key: [u8; N]) -> Result<(), Error> {
        match &mut self.spill {
            None => {
                self.keys.push(key);
                Ok(())
            }
            Some(spill) => Ok(spill.insert(&key)?),
        }
    }

    /// Sort and deduplicate the inserted keys by their first `prefix_len` bytes and produce the
    /// finished set.
    ///
    /// In-memory builders sort in place and keep the result in an owned buffer. External
    /// builders ([`Self::with_max_bucket_bytes`]) instead stream the spilled keys through a
    /// recursive most-significant-byte partition, sort one bucket at a time, and memory-map the
    /// finished set from an anonymous temporary file.
    ///
    /// # Arguments
    ///
    /// * `prefix_len` - How many leading bytes of each key to keep (`P`). Must be within
    ///   `3..=N`; lookups are exact when `prefix_len == N`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidPrefixLen`] if `prefix_len` is outside `3..=N`,
    /// [`Error::TooManyEntries`] if the deduplicated entry count overflows the `u32` offset
    /// table, and [`Error::Io`] if an external build cannot read or write its temporary files.
    pub fn build(mut self, prefix_len: usize) -> Result<ByteStringSet<N>, Error> {
        if !(OFFSET_TABLE_PREFIX_LEN..=N).contains(&prefix_len) {
            Err(Error::InvalidPrefixLen {
                prefix_len,
                key_len: N,
            })
        } else if let Some(spill) = self.spill {
            spill.build(prefix_len)
        } else {
            // Capacity is computed before deduplication, so this can over-allocate, matching
            // the previous behavior of reserving one stride per inserted key.
            let mut data = Vec::with_capacity(self.keys.len() * prefix_len);
            sort_dedup_emit(&mut self.keys, prefix_len, &mut data)?;
            let offsets = build_offsets(&data, prefix_len)?;
            Ok(ByteStringSet {
                prefix_len,
                data: Storage::Owned(data),
                offsets,
            })
        }
    }
}

/// A byte-string set over `[u8; N]` keys, matching on the first `P` bytes.
///
/// Created by [`ByteStringSetBuilder::build`] or loaded from a data file with
/// [`ByteStringSet::read`]. See the crate docs for the layout.
#[derive(Debug)]
pub struct ByteStringSet<const N: usize> {
    prefix_len: usize,
    data: Storage,
    offsets: Vec<u32>,
}

impl<const N: usize> ByteStringSet<N> {
    /// Create an empty [`ByteStringSetBuilder`] for assembling a set from `[u8; N]` keys.
    #[must_use]
    pub fn builder() -> ByteStringSetBuilder<N> {
        ByteStringSetBuilder::default()
    }

    /// Load a set from a data file previously written by [`write`].
    ///
    /// The file is memory-mapped rather than read into memory. Its header records the prefix
    /// length `P`, and the offset table (which is never serialized) is rebuilt by scanning the
    /// mapped data, which also validates that entries are sorted and deduplicated. The file must
    /// not be modified while the returned set is alive: mapped memory changing underneath the
    /// process is undefined behavior, which the operating system cannot guard against.
    ///
    /// # Arguments
    ///
    /// * `path` - Data file written by [`write`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the file cannot be opened or mapped, [`Error::BadHeader`] if it
    /// is truncated or lacks the header magic, [`Error::UnsupportedVersion`] for an unknown
    /// format version, [`Error::InvalidPrefixLen`] if the recorded prefix length is outside
    /// `3..=N`, [`Error::Misaligned`] if the data size is not a multiple of the prefix length,
    /// [`Error::Unsorted`] if entries are not strictly ascending, and [`Error::TooManyEntries`]
    /// if the entry count overflows `u32`.
    ///
    /// [`write`]: ByteStringSet::write
    pub fn read(path: impl AsRef<Path>) -> Result<Self, Error> {
        let file = File::open(path)?;
        // SAFETY: mapping a file is only unsound if another process truncates or rewrites it
        // while mapped; callers must uphold the immutability requirement documented above.
        let map = unsafe { Mmap::map(&file)? };
        // `split_first_chunk` returns the leading bytes as a fixed-size `&[u8; HEADER_LEN]`
        // array (plus the rest of the map), or `None` if the map is too short.
        let Some((header, data)) = map.split_first_chunk::<HEADER_LEN>() else {
            return Err(Error::BadHeader);
        };
        // `prefix_len_bytes @ ..` binds the trailing 8 bytes as a `[u8; 8]`.
        let &[
            magic0,
            magic1,
            magic2,
            magic3,
            version,
            _,
            _,
            _,
            prefix_len_bytes @ ..,
        ] = header;

        if [magic0, magic1, magic2, magic3] != MAGIC {
            Err(Error::BadHeader)
        } else if version != FORMAT_VERSION {
            Err(Error::UnsupportedVersion { version })
        } else {
            let raw_prefix_len = u64::from_le_bytes(prefix_len_bytes);
            let prefix_len = usize::try_from(raw_prefix_len).map_err(|_| Error::BadHeader)?;
            if (OFFSET_TABLE_PREFIX_LEN..=N).contains(&prefix_len) {
                if data.len() % prefix_len != 0 {
                    Err(Error::Misaligned {
                        data_len: data.len(),
                        prefix_len,
                    })
                } else {
                    let offsets = build_offsets(data, prefix_len)?;
                    Ok(Self {
                        prefix_len,
                        data: Storage::Mapped(map),
                        offsets,
                    })
                }
            } else {
                Err(Error::InvalidPrefixLen {
                    prefix_len,
                    key_len: N,
                })
            }
        }
    }

    /// Test whether a key's first `P` bytes are in the set.
    ///
    /// Exact membership when `prefix_len == N`; otherwise true if any member shares the key's
    /// `P`-byte prefix.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to probe.
    ///
    /// # Returns
    ///
    /// `true` if the key's `P`-byte prefix is present.
    #[must_use]
    pub fn lookup(&self, key: [u8; N]) -> bool {
        let probe = &key[..self.prefix_len];
        let data = self.data.as_slice();
        // The first 3 bytes select a bucket; binary-search the rest of it.
        let bucket = table_index(probe);
        let mut low = self.offsets[bucket] as usize;
        let mut high = self.offsets[bucket + 1] as usize;
        while low < high {
            let middle = low + (high - low) / 2;
            let start = middle * self.prefix_len;
            match data[start..start + self.prefix_len].cmp(probe) {
                Ordering::Less => low = middle + 1,
                Ordering::Greater => high = middle,
                Ordering::Equal => return true,
            }
        }
        false
    }

    /// Number of distinct `P`-byte prefixes stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.as_slice().len() / self.prefix_len
    }

    /// Whether the set contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.as_slice().is_empty()
    }

    /// The prefix length `P` this set matches on.
    #[must_use]
    pub const fn prefix_len(&self) -> usize {
        self.prefix_len
    }

    /// Write the set to a file loadable by [`read`].
    ///
    /// The file is a small header (magic, format version, and the prefix length `P`) followed by
    /// the packed prefix strides; the offset table is reconstructed on load.
    ///
    /// # Arguments
    ///
    /// * `path` - Destination file; created or truncated.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if the file cannot be created or written; IO is the only way this
    /// operation can fail.
    ///
    /// [`read`]: ByteStringSet::read
    pub fn write(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut file = File::create(path)?;
        write_header(&mut file, self.prefix_len)?;
        file.write_all(self.data.as_slice())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compose a data file image: a valid header followed by `payload`.
    fn raw_file(prefix_len: u64, payload: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0u8; HEADER_LEN];
        bytes[..4].copy_from_slice(&MAGIC);
        bytes[4] = FORMAT_VERSION;
        bytes[8..16].copy_from_slice(&prefix_len.to_le_bytes());
        bytes.extend_from_slice(payload);
        bytes
    }

    fn sample_set() -> ByteStringSet<4> {
        let mut builder = ByteStringSet::<4>::builder();
        for key in [*b"abcd", *b"abce", *b"zzzz", *b"\x00\x00\x00\x01"] {
            builder.insert(key).expect("insert key");
        }
        builder.build(4).expect("valid build")
    }

    /// Build a set with the external-sort builder from `keys`.
    fn external_set<const N: usize>(
        max_bucket_bytes: usize,
        keys: &[[u8; N]],
        prefix_len: usize,
    ) -> ByteStringSet<N> {
        let mut builder = ByteStringSetBuilder::<N>::with_max_bucket_bytes(max_bucket_bytes)
            .expect("create spill file");
        for &key in keys {
            builder.insert(key).expect("spill key");
        }
        builder.build(prefix_len).expect("valid build")
    }

    #[test]
    fn exact_lookup_hits_and_misses() {
        let set = sample_set();
        assert!(set.lookup(*b"abcd"));
        assert!(set.lookup(*b"abce"));
        assert!(set.lookup(*b"zzzz"));
        assert!(set.lookup(*b"\x00\x00\x00\x01"));
        assert!(!set.lookup(*b"abcf"));
        assert!(!set.lookup(*b"\x00\x00\x00\x00"));
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn prefix_lookup_matches_shared_prefixes() {
        let mut builder = ByteStringSet::<8>::builder();
        builder.insert(*b"abcd1234").expect("insert key");
        builder.insert(*b"wxyz0000").expect("insert key");
        let set = builder.build(4).expect("valid build");
        // Only the first 4 bytes are kept, so any suffix matches.
        assert!(set.lookup(*b"abcdXXXX"));
        assert!(set.lookup(*b"wxyz9999"));
        assert!(!set.lookup(*b"abzz1234"));
    }

    #[test]
    fn duplicates_and_shared_prefixes_are_deduped() {
        let mut builder = ByteStringSet::<8>::builder();
        builder.insert(*b"abcd1234").expect("insert key");
        builder.insert(*b"abcd1234").expect("insert key");
        builder.insert(*b"abcd9999").expect("insert key");
        let set = builder.build(4).expect("valid build");
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn empty_set_misses_everything() {
        let set = ByteStringSet::<4>::builder().build(4).expect("valid build");
        assert!(set.is_empty());
        assert!(!set.lookup(*b"abcd"));
    }

    #[test]
    fn prefix_len_bounds_are_enforced() {
        assert!(matches!(
            ByteStringSet::<4>::builder().build(2),
            Err(Error::InvalidPrefixLen {
                prefix_len: 2,
                key_len: 4
            })
        ));
        assert!(matches!(
            ByteStringSet::<4>::builder().build(5),
            Err(Error::InvalidPrefixLen {
                prefix_len: 5,
                key_len: 4
            })
        ));
    }

    #[test]
    fn external_build_single_bucket() {
        // A huge budget keeps the whole spill file in one bucket, exercising the load-and-sort
        // path without any partitioning.
        let set = external_set(usize::MAX, &[*b"abcd", *b"abce", *b"zzzz", *b"abcd"], 4);
        assert_eq!(set.len(), 3);
        assert!(set.lookup(*b"abcd"));
        assert!(set.lookup(*b"abce"));
        assert!(set.lookup(*b"zzzz"));
        assert!(!set.lookup(*b"aaaa"));
    }

    #[test]
    fn external_build_partitions_oversized_buckets() {
        // A one-key budget forces partitioning down to single-key buckets and, for the repeated
        // "aaaa" keys, all the way to the shared-prefix early exit at `depth == prefix_len`.
        let keys = [
            *b"aaaa", *b"aaab", *b"aaba", *b"abaa", *b"baaa", *b"aaaa", *b"zzzz",
        ];
        let set = external_set(4, &keys, 4);
        assert_eq!(set.len(), 6);
        for &key in &keys {
            assert!(set.lookup(key));
        }
        assert!(!set.lookup(*b"aabb"));
    }

    #[test]
    fn external_build_dedupes_shared_prefixes() {
        // Keys sharing a 4-byte prefix collapse to one entry even when partitioning separates
        // sorting into single-key buckets.
        let keys = [*b"abcd1234", *b"abcd9999", *b"wxyz0000"];
        let set = external_set(8, &keys, 4);
        assert_eq!(set.len(), 2);
        assert!(set.lookup(*b"abcdXXXX"));
        assert!(set.lookup(*b"wxyz9999"));
        assert!(!set.lookup(*b"abzz0000"));
    }

    #[test]
    fn external_build_empty() {
        let set = external_set::<4>(1024, &[], 4);
        assert!(set.is_empty());
        assert!(!set.lookup(*b"abcd"));
    }

    #[test]
    fn external_build_roundtrips_through_file() {
        let set = external_set(4, &[*b"abcd", *b"zzzz"], 4);
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("set.bin");
        set.write(&path).expect("write data file");

        let loaded = ByteStringSet::<4>::read(&path).expect("load data file");
        assert_eq!(loaded.len(), 2);
        assert!(loaded.lookup(*b"abcd"));
        assert!(loaded.lookup(*b"zzzz"));
        assert!(!loaded.lookup(*b"abce"));
    }

    #[test]
    fn file_roundtrip_preserves_membership() {
        let set = sample_set();
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("set.bin");
        set.write(&path).expect("write data file");

        let loaded = ByteStringSet::<4>::read(&path).expect("load data file");
        assert_eq!(loaded.prefix_len(), 4);
        assert_eq!(loaded.len(), set.len());
        assert!(loaded.lookup(*b"abcd"));
        assert!(loaded.lookup(*b"zzzz"));
        assert!(!loaded.lookup(*b"abcf"));
    }

    #[test]
    fn read_rejects_misaligned_files() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("bad.bin");
        std::fs::write(&path, raw_file(4, b"abcde")).expect("write file");
        assert!(matches!(
            ByteStringSet::<4>::read(&path),
            Err(Error::Misaligned {
                data_len: 5,
                prefix_len: 4
            })
        ));
    }

    #[test]
    fn read_rejects_unsorted_files() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("unsorted.bin");
        std::fs::write(&path, raw_file(4, b"zzzzaaaa")).expect("write file");
        assert!(matches!(
            ByteStringSet::<4>::read(&path),
            Err(Error::Unsorted { index: 1 })
        ));
    }

    #[test]
    fn read_rejects_missing_or_truncated_header() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("garbage.bin");
        std::fs::write(&path, b"not a bsset file").expect("write file");
        assert!(matches!(
            ByteStringSet::<4>::read(&path),
            Err(Error::BadHeader)
        ));

        std::fs::write(&path, b"BSST").expect("write file");
        assert!(matches!(
            ByteStringSet::<4>::read(&path),
            Err(Error::BadHeader)
        ));
    }

    #[test]
    fn read_rejects_unsupported_version() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("future.bin");
        let mut bytes = raw_file(4, b"abcd");
        bytes[4] = 2;
        std::fs::write(&path, bytes).expect("write file");
        assert!(matches!(
            ByteStringSet::<4>::read(&path),
            Err(Error::UnsupportedVersion { version: 2 })
        ));
    }

    #[test]
    fn read_rejects_prefix_exceeding_key_len() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("wide.bin");
        std::fs::write(&path, raw_file(5, b"abcde")).expect("write file");
        assert!(matches!(
            ByteStringSet::<4>::read(&path),
            Err(Error::InvalidPrefixLen {
                prefix_len: 5,
                key_len: 4
            })
        ));
    }
}

#[cfg(test)]
mod proptests {
    use std::collections::BTreeSet;

    use proptest::prelude::*;

    use super::*;

    const KEY_LEN: usize = 8;

    /// Keys drawn from the full byte range or from a tiny alphabet; the latter forces shared
    /// prefixes, duplicate keys, and crowded offset-table buckets, which uniform random bytes
    /// almost never produce.
    fn key_strategy() -> impl Strategy<Value = [u8; KEY_LEN]> {
        prop_oneof![
            proptest::array::uniform8(any::<u8>()),
            proptest::array::uniform8(0u8..4),
        ]
    }

    /// Reference model: the distinct `prefix_len`-byte prefixes of `keys`.
    fn reference_prefixes(keys: &[[u8; KEY_LEN]], prefix_len: usize) -> BTreeSet<Vec<u8>> {
        keys.iter().map(|key| key[..prefix_len].to_vec()).collect()
    }

    fn build_set(keys: &[[u8; KEY_LEN]], prefix_len: usize) -> ByteStringSet<KEY_LEN> {
        let mut builder = ByteStringSet::<KEY_LEN>::builder();
        for &key in keys {
            builder.insert(key).expect("in-memory insert cannot fail");
        }
        builder
            .build(prefix_len)
            .expect("prefix_len is within 3..=KEY_LEN")
    }

    proptest! {
        // Each case allocates the 64 MiB offset table, so keep the case count moderate.
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn lookup_matches_reference_model(
            keys in proptest::collection::vec(key_strategy(), 0..64),
            probes in proptest::collection::vec(key_strategy(), 0..64),
            prefix_len in OFFSET_TABLE_PREFIX_LEN..=KEY_LEN,
        ) {
            let set = build_set(&keys, prefix_len);
            let reference = reference_prefixes(&keys, prefix_len);
            prop_assert_eq!(set.len(), reference.len());
            // Inserted keys must all hit; probes must agree with the model either way.
            for &probe in keys.iter().chain(&probes) {
                prop_assert_eq!(set.lookup(probe), reference.contains(&probe[..prefix_len]));
            }
        }

        #[test]
        fn file_roundtrip_preserves_lookups(
            keys in proptest::collection::vec(key_strategy(), 0..64),
            probes in proptest::collection::vec(key_strategy(), 0..64),
            prefix_len in OFFSET_TABLE_PREFIX_LEN..=KEY_LEN,
        ) {
            let set = build_set(&keys, prefix_len);
            let dir = tempfile::tempdir().expect("create temp dir");
            let path = dir.path().join("set.bin");
            set.write(&path).expect("write data file");

            let loaded = ByteStringSet::<KEY_LEN>::read(&path).expect("load data file");
            prop_assert_eq!(loaded.prefix_len(), prefix_len);
            prop_assert_eq!(loaded.len(), set.len());
            for &probe in keys.iter().chain(&probes) {
                prop_assert_eq!(loaded.lookup(probe), set.lookup(probe));
            }
        }

        #[test]
        fn external_build_matches_in_memory(
            keys in proptest::collection::vec(key_strategy(), 0..64),
            probes in proptest::collection::vec(key_strategy(), 0..64),
            prefix_len in OFFSET_TABLE_PREFIX_LEN..=KEY_LEN,
            // Small budgets force deep recursive partitioning; the unit tests cover the
            // single-bucket case.
            max_bucket_bytes in 0usize..=128,
        ) {
            let in_memory = build_set(&keys, prefix_len);
            let mut builder = ByteStringSetBuilder::<KEY_LEN>::with_max_bucket_bytes(max_bucket_bytes)
                .expect("create spill file");
            for &key in &keys {
                builder.insert(key).expect("spill key");
            }
            let external = builder.build(prefix_len).expect("valid build");
            prop_assert_eq!(external.len(), in_memory.len());
            for &probe in keys.iter().chain(&probes) {
                prop_assert_eq!(external.lookup(probe), in_memory.lookup(probe));
            }
        }
    }
}
