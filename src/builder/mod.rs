//! Incremental construction of [`ByteStringSet`]s from fixed-length `[u8; N]` keys.

mod external;

use self::external::SpillState;
use crate::{ByteStringSet, Error, OFFSET_TABLE_PREFIX_LEN, Storage, build_offsets};
use std::io::{self, Write};

/// Sort `keys`, drop all but the first key of each distinct `prefix_len`-byte prefix, and write
/// the surviving prefixes to `out` in order.
///
/// Shared by the in-memory build and, bucket by bucket, the external build in [`external`].
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
/// Obtained from [`ByteStringSet::builder`] (in-memory) or [`ByteStringSet::external_builder`]
/// (external sort). Duplicate insertions are fine; they are removed during [`build`].
///
/// [`build`]: ByteStringSetBuilder::build
#[derive(Debug)]
pub struct ByteStringSetBuilder<const N: usize> {
    /// Keys buffered in memory; unused (always empty) when `spill` is active.
    keys: Vec<[u8; N]>,
    /// External-sort state; `None` for the in-memory builder.
    spill: Option<SpillState>,
}

impl<const N: usize> Default for ByteStringSetBuilder<N> {
    fn default() -> Self {
        Self::in_memory()
    }
}

impl<const N: usize> ByteStringSetBuilder<N> {
    /// Create an in-memory builder; the public entry point is [`ByteStringSet::builder`].
    pub(crate) const fn in_memory() -> Self {
        Self {
            keys: Vec::new(),
            spill: None,
        }
    }

    /// Create an external-sort builder; the public entry point is
    /// [`ByteStringSet::external_builder`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the spill file cannot be created.
    pub(crate) fn external(max_bucket_bytes: usize) -> Result<Self, Error> {
        Ok(Self {
            keys: Vec::new(),
            spill: Some(SpillState::new(max_bucket_bytes.max(N))?),
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
    /// builders (from [`ByteStringSet::external_builder`]) instead stream the spilled keys
    /// through a recursive most-significant-byte partition, sort one bucket at a time, and
    /// memory-map the finished set from an anonymous temporary file.
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
