//! Incremental construction of [`ByteStringSet`]s from fixed-length `[u8; N]` keys.

mod external;

use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;

use self::external::SpillState;
use crate::{
    ByteStringSet, Error, HEADER_LEN, OFFSET_TABLE_PREFIX_LEN, Storage, build_offsets, parse_header,
};

/// Sort and deduplicate keys by the prefix the finished set retains.
fn sort_dedup<const N: usize>(keys: &mut Vec<[u8; N]>, prefix_len: usize) {
    keys.sort_unstable_by(|left, right| left[..prefix_len].cmp(&right[..prefix_len]));
    keys.dedup_by(|next, kept| next[..prefix_len] == kept[..prefix_len]);
}

/// Write the prefixes of sorted, deduplicated keys in order.
fn emit<const N: usize>(
    keys: &[[u8; N]],
    prefix_len: usize,
    out: &mut impl Write,
) -> std::io::Result<()> {
    for key in keys {
        out.write_all(&key[..prefix_len])?;
    }
    Ok(())
}

/// Append `count` keys read from `reader` to `keys`, zero-padding each from its leading
/// `prefix_len` bytes.
fn read_keys_into<const N: usize>(
    reader: &mut impl Read,
    count: usize,
    prefix_len: usize,
    keys: &mut Vec<[u8; N]>,
) -> std::io::Result<()> {
    keys.try_reserve(count).map_err(std::io::Error::other)?;
    let original_len = keys.len();
    let mut key = [0u8; N];
    let result = (0..count).try_for_each(|_| {
        reader.read_exact(&mut key[..prefix_len])?;
        keys.push(key);
        Ok(())
    });
    if result.is_err() {
        keys.truncate(original_len);
    }
    result
}

/// Accumulates `[u8; N]` keys and builds a [`ByteStringSet`].
///
/// Obtained from [`ByteStringSet::builder`] (in-memory) or [`ByteStringSet::external_builder`]
/// (external sort). Duplicate insertions are fine; they are removed during [`build`], and
/// existing data files can be merged in with [`import`].
///
/// [`build`]: ByteStringSetBuilder::build
/// [`import`]: ByteStringSetBuilder::import
#[derive(Debug)]
pub struct ByteStringSetBuilder<const N: usize> {
    /// Keys buffered in memory; unused (always empty) when `spill` is active.
    keys: Vec<[u8; N]>,
    /// External-sort state; `None` for the in-memory builder.
    spill: Option<SpillState<N>>,
    /// The prefix length (`P`) the finished set will match on; range-validated by
    /// [`build`](Self::build) and [`import`](Self::import) rather than at construction.
    prefix_len: usize,
}

impl<const N: usize> Default for ByteStringSetBuilder<N> {
    /// An in-memory builder that matches on full keys (`prefix_len == N`).
    fn default() -> Self {
        Self::in_memory(N)
    }
}

impl<const N: usize> ByteStringSetBuilder<N> {
    /// Create an in-memory builder; the public entry point is [`ByteStringSet::builder`].
    pub(crate) const fn in_memory(prefix_len: usize) -> Self {
        Self {
            keys: Vec::new(),
            spill: None,
            prefix_len,
        }
    }

    /// Create an external-sort builder; the public entry point is
    /// [`ByteStringSet::external_builder`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the spill file cannot be created.
    pub(crate) fn external(prefix_len: usize, max_bucket_bytes: usize) -> Result<Self, Error> {
        Ok(Self {
            keys: Vec::new(),
            spill: Some(SpillState::new(max_bucket_bytes)?),
            prefix_len,
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
    /// [`ByteStringSet::builder`]) never fail. An external builder cannot be used after a
    /// failed write.
    pub fn insert(&mut self, key: [u8; N]) -> Result<(), Error> {
        match &mut self.spill {
            None => {
                self.keys.push(key);
                Ok(())
            }
            Some(spill) => Ok(spill.insert(&key)?),
        }
    }

    /// Read the entries of an existing data file into the builder.
    ///
    /// The file's prefix length must be at least the builder's: such entries truncate cleanly
    /// to the target prefix during [`build`], while shorter ones could not be reconstituted
    /// into keys. Entries shorter than `N` are zero-padded into full keys; the padding never
    /// reaches a finished set, because the builder's shorter-or-equal prefix length always
    /// truncates it away. The imported entries join the keys inserted so far, so several sets
    /// (plus loose keys) can be merged into one build. Import order does not matter: [`build`]
    /// re-sorts and deduplicates everything. The input file must remain unchanged during import.
    ///
    /// A failed in-memory import leaves the existing keys unchanged. An I/O failure while
    /// appending imported entries invalidates an external builder; subsequent inserts, imports,
    /// and builds return an I/O error. Failures detected before any entries are read (such as
    /// an invalid header or a misaligned file) leave either builder usable.
    ///
    /// # Arguments
    ///
    /// * `path` - Data file previously written by [`ByteStringSet::write`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the file cannot be opened or read, [`Error::BadHeader`] if it is
    /// truncated or lacks the header magic, [`Error::UnsupportedVersion`] for an unknown format
    /// version, [`Error::InvalidPrefixLen`] if the builder's or the file's prefix length is
    /// outside `3..=N`, [`Error::PrefixLenExceedsImport`] if the file's prefix length is
    /// shorter than the builder's, and [`Error::Misaligned`] if the data size is not a whole
    /// number of entries. Allocation failures while buffering an import return [`Error::Io`].
    ///
    /// [`build`]: ByteStringSetBuilder::build
    pub fn import(&mut self, path: impl AsRef<Path>) -> Result<(), Error> {
        if !(OFFSET_TABLE_PREFIX_LEN..=N).contains(&self.prefix_len) {
            return Err(Error::InvalidPrefixLen {
                prefix_len: self.prefix_len,
                key_len: N,
            });
        }
        let file = File::open(path)?;
        let Some(data_len) = file.metadata()?.len().checked_sub(HEADER_LEN as u64) else {
            return Err(Error::BadHeader);
        };
        let mut reader = BufReader::new(file);
        let mut header = [0u8; HEADER_LEN];
        reader.read_exact(&mut header)?;
        let imported_prefix_len = parse_header(&header)?;
        if !(OFFSET_TABLE_PREFIX_LEN..=N).contains(&imported_prefix_len) {
            return Err(Error::InvalidPrefixLen {
                prefix_len: imported_prefix_len,
                key_len: N,
            });
        }
        if imported_prefix_len < self.prefix_len {
            return Err(Error::PrefixLenExceedsImport {
                prefix_len: self.prefix_len,
                imported_prefix_len,
            });
        }
        if data_len % imported_prefix_len as u64 != 0 {
            return Err(Error::Misaligned {
                // The saturating conversion only affects the error report, and only on targets
                // where `usize` is narrower than `u64`.
                data_len: usize::try_from(data_len).unwrap_or(usize::MAX),
                prefix_len: imported_prefix_len,
            });
        }
        let count = data_len / imported_prefix_len as u64;
        if let Some(spill) = &mut self.spill {
            spill.import(&mut reader, count, imported_prefix_len)?;
        } else {
            let count = usize::try_from(count)
                .map_err(|_| std::io::Error::other("data file too large to buffer in memory"))?;
            read_keys_into(&mut reader, count, imported_prefix_len, &mut self.keys)?;
        }
        Ok(())
    }

    /// Sort and deduplicate the inserted keys by their first `prefix_len` bytes (as given at
    /// construction) and produce the finished set.
    ///
    /// In-memory builders sort in place and keep the result in an owned buffer. External
    /// builders (from [`ByteStringSet::external_builder`]) instead stream the spilled keys
    /// through most-significant-byte partitioning, sort one bucket at a time, and
    /// memory-map the finished set from an anonymous temporary file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidPrefixLen`] if the builder's prefix length is outside `3..=N`,
    /// [`Error::TooManyEntries`] if the deduplicated entry count overflows the `u32` offset
    /// table, and [`Error::Io`] if an external build cannot read or write its temporary files
    /// or was invalidated by a failed insert or import.
    pub fn build(mut self) -> Result<ByteStringSet<N>, Error> {
        let prefix_len = self.prefix_len;
        if !(OFFSET_TABLE_PREFIX_LEN..=N).contains(&prefix_len) {
            Err(Error::InvalidPrefixLen {
                prefix_len,
                key_len: N,
            })
        } else if let Some(spill) = self.spill {
            spill.build(prefix_len)
        } else {
            sort_dedup(&mut self.keys, prefix_len);
            let mut data = Vec::with_capacity(self.keys.len() * prefix_len);
            emit(&self.keys, prefix_len, &mut data)?;
            // Release the full keys before allocating the 64 MiB offset table.
            drop(self.keys);
            let offsets = build_offsets(&data, prefix_len)?;
            Ok(ByteStringSet {
                prefix_len,
                data: Storage::Owned(data),
                offsets,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncated_import_preserves_existing_keys() {
        let mut keys = vec![*b"keep"];
        let error = read_keys_into(&mut b"abcdx".as_slice(), 2, 4, &mut keys)
            .expect_err("second key is truncated");
        assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof);
        assert_eq!(keys, [*b"keep"]);
    }

    #[test]
    fn impossible_import_capacity_returns_an_error() {
        let mut keys = vec![*b"keep"];
        assert!(read_keys_into(&mut std::io::empty(), usize::MAX, 4, &mut keys).is_err());
        assert_eq!(keys, [*b"keep"]);
    }
}
