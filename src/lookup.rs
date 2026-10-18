use std::borrow::Borrow;
use std::cmp::Ordering;

use super::{ByteStringSet, table_index};

impl<const N: usize> ByteStringSet<N> {
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
        let bucket = table_index(probe);
        self.search(
            probe,
            self.offsets[bucket] as usize,
            self.offsets[bucket + 1] as usize,
        )
        .is_ok()
    }

    /// Test each key's first `P` bytes, yielding results in input order.
    ///
    /// Accepts owned or borrowed `[u8; N]` keys and has the same membership semantics as
    /// [`lookup`](Self::lookup). The iterator consumes one key per result without buffering
    /// the batch. Inputs need not be sorted or deduplicated. For sorted inputs, use
    /// [`lookup_all_sorted`](Self::lookup_all_sorted) to reuse search positions.
    ///
    /// # Examples
    ///
    /// ```
    /// use bsset::ByteStringSet;
    ///
    /// let mut builder = ByteStringSet::<4>::builder();
    /// builder.insert(*b"abcd")?;
    /// let set = builder.build()?;
    /// let keys = [*b"zzzz", *b"abcd", *b"abcd"];
    /// assert_eq!(set.lookup_all(&keys).collect::<Vec<_>>(), [false, true, true]);
    /// # Ok::<(), bsset::Error>(())
    /// ```
    pub fn lookup_all<I: IntoIterator<Item: Borrow<[u8; N]>>>(
        &self,
        keys: I,
    ) -> impl Iterator<Item = bool> {
        keys.into_iter().map(|key| self.lookup(*key.borrow()))
    }

    /// Test sorted keys' first `P` bytes, yielding results in input order.
    ///
    /// Like [`lookup_all`](Self::lookup_all), but the keys' `P`-byte prefixes must be in
    /// nondecreasing lexicographic order. Sorting full keys satisfies this requirement;
    /// duplicates are allowed, and bytes after the prefix do not affect the required order.
    /// The iterator consumes one key per result without buffering the batch, retaining the
    /// previous key to check ordering.
    ///
    /// Within each three-byte bucket, probes reuse the previous search position and use
    /// exponential search to skip gaps. Nearby probes and repeated prefixes take constant
    /// time; larger gaps take logarithmic time in the number of skipped entries. The first
    /// probe in each bucket uses binary search. The benefit is greatest when many probes
    /// share a bucket; for small buckets, checking order can outweigh the savings.
    ///
    /// # Panics
    ///
    /// Advancing the iterator panics if a key's `P`-byte prefix is less than the previous one.
    ///
    /// # Examples
    ///
    /// ```
    /// use bsset::ByteStringSet;
    ///
    /// let mut builder = ByteStringSet::<4>::builder();
    /// builder.insert(*b"abcd")?;
    /// builder.insert(*b"abcf")?;
    /// let set = builder.build()?;
    /// let keys = [*b"abcd", *b"abce", *b"abcf"];
    /// assert_eq!(set.lookup_all_sorted(&keys).collect::<Vec<_>>(), [true, false, true]);
    /// # Ok::<(), bsset::Error>(())
    /// ```
    pub fn lookup_all_sorted<I: IntoIterator<Item: Borrow<[u8; N]>>>(
        &self,
        keys: I,
    ) -> impl Iterator<Item = bool> {
        let mut previous: Option<I::Item> = None;
        let mut previous_bucket = 0;
        let mut position = 0;
        keys.into_iter().map(move |key| {
            let probe = &key.borrow()[..self.prefix_len];
            assert!(
                previous
                    .as_ref()
                    .is_none_or(|previous| &previous.borrow()[..self.prefix_len] <= probe),
                "lookup_all_sorted requires nondecreasing key prefixes"
            );
            let bucket = table_index(probe);
            let high = self.offsets[bucket + 1] as usize;
            let result = if bucket == previous_bucket && previous.is_some() {
                self.search_forward(probe, position, high)
            } else {
                self.search(probe, self.offsets[bucket] as usize, high)
            };
            position = match result {
                Ok(index) | Err(index) => index,
            };
            previous = Some(key);
            previous_bucket = bucket;
            result.is_ok()
        })
    }

    /// Find a prefix or its insertion position within an entry range.
    fn search(&self, probe: &[u8], mut low: usize, mut high: usize) -> Result<usize, usize> {
        let data = self.data.as_slice();
        while low < high {
            let middle = low + (high - low) / 2;
            let start = middle * self.prefix_len;
            match data[start..start + self.prefix_len].cmp(probe) {
                Ordering::Less => low = middle + 1,
                Ordering::Greater => high = middle,
                Ordering::Equal => return Ok(middle),
            }
        }
        Err(low)
    }

    /// Bound the next search exponentially so nearby probes need few comparisons without
    /// scanning every intervening entry when probes are far apart.
    fn search_forward(&self, probe: &[u8], mut low: usize, high: usize) -> Result<usize, usize> {
        let data = self.data.as_slice();
        let mut step = 1usize;
        while low < high {
            let middle = low + (step - 1).min(high - low - 1);
            let start = middle * self.prefix_len;
            match data[start..start + self.prefix_len].cmp(probe) {
                Ordering::Less => {
                    low = middle + 1;
                    step = step.saturating_mul(2);
                }
                Ordering::Greater => return self.search(probe, low, middle),
                Ordering::Equal => return Ok(middle),
            }
        }
        Err(low)
    }
}
