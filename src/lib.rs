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
//! Building normally buffers all keys in memory. For sets that may not fit in memory,
//! [`ByteStringSet::external_builder`] spills keys to temporary files and builds the set with an
//! external most-significant-byte radix sort instead.
//!
//! # Examples
//!
//! ```
//! use bsset::ByteStringSet;
//!
//! // Prefix length 4 == key length 4, so lookups are exact.
//! let mut builder = ByteStringSet::<4>::builder();
//! builder.insert(*b"abcd")?;
//! builder.insert(*b"wxyz")?;
//!
//! let set = builder.build()?;
//! assert!(set.lookup(*b"abcd"));
//! assert!(!set.lookup(*b"aaaa"));
//! # Ok::<(), bsset::Error>(())
//! ```

pub mod builder;

use std::cmp::Ordering;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use builder::ByteStringSetBuilder;
use memmap2::Mmap;
use thiserror::Error;

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
    /// A data file with a prefix length shorter than the builder's cannot be imported: its
    /// entries carry no information past their original length.
    #[error("prefix length {prefix_len} exceeds imported prefix length {imported_prefix_len}")]
    PrefixLenExceedsImport {
        /// The builder's prefix length.
        prefix_len: usize,
        /// The prefix length recorded in the data file being imported.
        imported_prefix_len: usize,
    },
    /// An underlying filesystem operation failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
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
const fn table_index(prefix: &[u8]) -> usize {
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

    let count = u32::try_from(count).map_err(|_| Error::TooManyEntries { count })?;
    let mut offsets = Vec::with_capacity(OFFSET_TABLE_LEN);
    let mut previous: Option<&[u8]> = None;
    for (index, entry) in (0..count).zip(data.chunks_exact(prefix_len)) {
        if previous.is_some_and(|previous| previous >= entry) {
            return Err(Error::Unsorted {
                index: index as usize,
            });
        }
        // Sorted entries let us fill empty buckets and the next occupied bucket directly.
        // Repeated entries in the same bucket leave its start offset unchanged.
        offsets.resize(table_index(entry) + 1, index);
        previous = Some(entry);
    }
    offsets.resize(OFFSET_TABLE_LEN, count);
    Ok(offsets)
}

/// Write the data file header (magic, format version, and prefix length) to `out`.
fn write_header(out: &mut impl Write, prefix_len: usize) -> std::io::Result<()> {
    let mut header = [0u8; HEADER_LEN];
    header[..4].copy_from_slice(&MAGIC);
    header[4] = FORMAT_VERSION;
    header[8..16].copy_from_slice(&(prefix_len as u64).to_le_bytes());
    out.write_all(&header)
}

/// Parse a data file header, returning the recorded prefix length.
///
/// Validates the magic bytes and format version; range-checking the prefix length is left to
/// the caller, which knows its key length `N`.
fn parse_header(header: &[u8; HEADER_LEN]) -> Result<usize, Error> {
    // Destructuring a fixed-size array cannot fail, so no `try_into`/`expect` is needed;
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
        usize::try_from(u64::from_le_bytes(prefix_len_bytes)).map_err(|_| Error::BadHeader)
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
    /// Create an empty builder that assembles a set from `[u8; N]` keys in memory.
    ///
    /// The set keeps full keys (`prefix_len == N`), so lookups are exact; use
    /// [`builder_with_prefix_len`] to match on a shorter prefix instead. Inserted keys are
    /// buffered in a `Vec` and sorted during [`build`], so the full key set must fit in memory;
    /// use [`external_builder`] for sets that may not.
    ///
    /// [`build`]: ByteStringSetBuilder::build
    /// [`builder_with_prefix_len`]: ByteStringSet::builder_with_prefix_len
    /// [`external_builder`]: ByteStringSet::external_builder
    #[must_use]
    pub const fn builder() -> ByteStringSetBuilder<N> {
        ByteStringSetBuilder::in_memory(N)
    }

    /// Create an empty in-memory builder, like [`builder`], for a set that matches on a chosen
    /// number of leading key bytes.
    ///
    /// # Arguments
    ///
    /// * `prefix_len` - How many leading bytes of each key the set keeps and matches on (`P`).
    ///   Must be within `3..=N`, which [`build`] and [`import`] check; lookups are exact when
    ///   `prefix_len == N`.
    ///
    /// [`build`]: ByteStringSetBuilder::build
    /// [`builder`]: ByteStringSet::builder
    /// [`import`]: ByteStringSetBuilder::import
    #[must_use]
    pub const fn builder_with_prefix_len(prefix_len: usize) -> ByteStringSetBuilder<N> {
        ByteStringSetBuilder::in_memory(prefix_len)
    }

    /// Create an empty builder that uses an external most-significant-byte radix sort so the
    /// full key set never has to fit in memory.
    ///
    /// Inserted keys stream to an anonymous spill file, and [`build`] partitions
    /// them by successive leading bytes until each bucket is at most `max_bucket_bytes`, sorts
    /// one bucket at a time in memory, and streams the deduplicated prefixes to a temporary data
    /// file that backs the finished set as a memory map. Peak memory during the build is
    /// therefore roughly `max_bucket_bytes` plus the fixed offset table and I/O buffers.
    /// Partitioning may open up to 256 temporary files at once. Shared prefix bytes are skipped,
    /// and each input partition is deleted before its children are processed.
    ///
    /// Temporary files are created in the system temporary directory (honoring the `TMPDIR`
    /// environment variable); if that is a RAM-backed filesystem such as a `tmpfs` `/tmp`, point
    /// `TMPDIR` at a disk-backed directory to get the full benefit. Values of `max_bucket_bytes`
    /// smaller than `N` are treated as `N`, since a bucket must hold at least one key.
    ///
    /// # Arguments
    ///
    /// * `prefix_len` - How many leading bytes of each key the set keeps and matches on (`P`).
    ///   Must be within `3..=N`, which [`build`] and [`import`] check; lookups are exact when
    ///   `prefix_len == N`.
    /// * `max_bucket_bytes` - Largest bucket of raw `N`-byte keys, in bytes, that [`build`] may
    ///   load and sort in memory at once.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the spill file cannot be created.
    ///
    /// [`build`]: ByteStringSetBuilder::build
    /// [`import`]: ByteStringSetBuilder::import
    pub fn external_builder(
        prefix_len: usize,
        max_bucket_bytes: usize,
    ) -> Result<ByteStringSetBuilder<N>, Error> {
        ByteStringSetBuilder::external(prefix_len, max_bucket_bytes)
    }

    /// Load a set from a data file previously written by [`write`].
    ///
    /// The file is memory-mapped rather than read into memory. Its header records the prefix
    /// length `P`, and the offset table (which is never serialized) is rebuilt by scanning the
    /// mapped data, which also validates that entries are sorted and deduplicated.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the underlying file is not modified or truncated, by this
    /// process or any other, from the start of this call until the returned set is dropped.
    /// Replacing a directory entry without modifying the mapped file is safe.
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
    #[expect(unsafe_code, reason = "mapping a file the caller keeps unchanged")]
    pub unsafe fn read(path: impl AsRef<Path>) -> Result<Self, Error> {
        let file = File::open(path)?;
        if file.metadata()?.len() < HEADER_LEN as u64 {
            return Err(Error::BadHeader);
        }
        // SAFETY: the caller guarantees that the file remains unchanged until the set is dropped.
        let map = unsafe { Mmap::map(&file)? };
        // `split_first_chunk` returns the leading bytes as a fixed-size `&[u8; HEADER_LEN]`
        // array (plus the rest of the map), or `None` if the map is too short.
        let Some((header, data)) = map.split_first_chunk::<HEADER_LEN>() else {
            return Err(Error::BadHeader);
        };
        let prefix_len = parse_header(header)?;
        if !(OFFSET_TABLE_PREFIX_LEN..=N).contains(&prefix_len) {
            Err(Error::InvalidPrefixLen {
                prefix_len,
                key_len: N,
            })
        } else if data.len() % prefix_len != 0 {
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
    /// A temporary file in the destination directory is written in full and synchronized to
    /// storage, then atomically replaces the destination, so a crash cannot leave a partially
    /// written file there. The directory is not synchronized, so after a crash the destination
    /// may still hold its previous contents. Existing memory maps and hard links keep their
    /// original data, and a destination symlink is replaced rather than followed. Some
    /// platforms may reject replacing a mapped file, in which case the existing file is left
    /// intact.
    ///
    /// The directory must be writable. On Unix the new file gets the permissions of any newly
    /// created file (`0o666` less the umask); an existing file's permissions are not preserved.
    ///
    /// # Arguments
    ///
    /// * `path` - Destination file; created or replaced.
    ///
    /// # Errors
    ///
    /// Returns [`std::io::Error`] if the file cannot be created or written; IO is the only way this
    /// operation can fail.
    ///
    /// [`read`]: ByteStringSet::read
    pub fn write(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let path = path.as_ref();
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let mut builder = tempfile::Builder::new();
        // tempfile defaults to owner-only permissions; match `File::create` instead.
        #[cfg(unix)]
        builder.permissions(std::os::unix::fs::PermissionsExt::from_mode(0o666));
        let mut file = builder.tempfile_in(parent)?;
        write_header(&mut file, self.prefix_len)?;
        file.write_all(self.data.as_slice())?;
        file.as_file().sync_all()?;
        file.persist(path).map_err(|error| error.error)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
