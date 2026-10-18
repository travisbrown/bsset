use std::cell::Cell;

use super::*;

#[test]
fn batch_lookup_preserves_order_and_duplicates() {
    let set = sample_set();
    // Revisit buckets after exhausting them, and move backwards within a bucket.
    let probes = [
        *b"abcf",
        *b"abce",
        *b"abce",
        *b"abca",
        *b"abcd",
        *b"abcf",
        *b"zzzz",
        *b"zzzy",
        *b"zzzz",
        *b"abcd",
        [0; 4],
        [255; 4],
        [0, 0, 0, 1],
    ];
    let expected: Vec<_> = probes.iter().map(|&key| set.lookup(key)).collect();
    assert_eq!(set.lookup_all(probes).collect::<Vec<_>>(), expected);
    assert_eq!(set.lookup_all(&probes).collect::<Vec<_>>(), expected);
    let mut probes = probes;
    probes.sort_unstable();
    let expected: Vec<_> = probes.iter().map(|&key| set.lookup(key)).collect();
    assert_eq!(set.lookup_all_sorted(probes).collect::<Vec<_>>(), expected);
    assert_eq!(set.lookup_all_sorted(&probes).collect::<Vec<_>>(), expected);
}

#[test]
fn batch_lookup_is_lazy() {
    let set = sample_set();
    let consumed = Cell::new(0);
    let keys = std::iter::repeat(*b"abcd").inspect(|_| consumed.set(consumed.get() + 1));
    let mut results = set.lookup_all(keys);
    assert_eq!(consumed.get(), 0);
    assert_eq!(results.next(), Some(true));
    assert_eq!(consumed.get(), 1);
    assert_eq!(results.take(3).collect::<Vec<_>>(), [true; 3]);
    assert_eq!(consumed.get(), 4);
    assert_eq!(set.lookup_all(std::iter::empty::<[u8; 4]>()).next(), None);
    let keys = std::iter::repeat(*b"abcd").inspect(|_| consumed.set(consumed.get() + 1));
    let mut results = set.lookup_all_sorted(keys);
    assert_eq!(consumed.get(), 4);
    assert_eq!(results.next(), Some(true));
    assert_eq!(consumed.get(), 5);
    assert_eq!(results.take(3).collect::<Vec<_>>(), [true; 3]);
    assert_eq!(consumed.get(), 8);
    assert_eq!(
        set.lookup_all_sorted(std::iter::empty::<[u8; 4]>()).next(),
        None
    );
}

#[test]
fn batch_lookup_of_empty_set() {
    let set = ByteStringSet::<4>::builder().build().expect("valid build");
    let probes = [*b"abca", *b"abca", *b"abcd", *b"aaaa", *b"zzzz"];
    assert_eq!(set.lookup_all(probes).collect::<Vec<_>>(), [false; 5]);
    let mut probes = probes;
    probes.sort_unstable();
    assert_eq!(
        set.lookup_all_sorted(probes).collect::<Vec<_>>(),
        [false; 5]
    );
}

#[test]
fn batch_lookup_ignores_suffix_order() {
    let keys = [*b"abcd1234", *b"abcf1234", *b"wxyz0000"];
    let probes = [
        *b"abcd9999",
        *b"abcd0000",
        *b"abce9999",
        *b"abce0000",
        *b"abcf0000",
        *b"wxyz9999",
    ];
    for prefix_len in [3, 4, 8] {
        let set = external_set(8, &keys, prefix_len);
        let expected: Vec<_> = probes.iter().map(|&key| set.lookup(key)).collect();
        assert_eq!(set.lookup_all(&probes).collect::<Vec<_>>(), expected);
        if prefix_len < 8 {
            assert_eq!(set.lookup_all_sorted(&probes).collect::<Vec<_>>(), expected);
        }
    }
}

#[test]
fn batch_lookup_handles_dense_and_sparse_probes() {
    // Keep all entries in one bucket, with misses before, between, and after stored prefixes.
    let key = |value: u32| {
        let bytes = value.to_be_bytes();
        [0, 0, 0, bytes[0], bytes[1], bytes[2], bytes[3], 0]
    };
    let mut builder = ByteStringSet::<8>::builder();
    for value in (2..8192).step_by(2) {
        builder.insert(key(value)).expect("insert key");
    }
    let set = builder.build().expect("valid build");
    for step in [1, 2, 3, 7, 31, 256, 1023, 4096] {
        let probes: Vec<_> = (0..=8192).step_by(step).map(key).collect();
        let expected: Vec<_> = probes.iter().map(|&probe| set.lookup(probe)).collect();
        assert_eq!(set.lookup_all(&probes).collect::<Vec<_>>(), expected);
        assert_eq!(set.lookup_all_sorted(&probes).collect::<Vec<_>>(), expected);
        assert_eq!(
            set.lookup_all(probes.iter().rev()).collect::<Vec<_>>(),
            expected.into_iter().rev().collect::<Vec<_>>()
        );
    }
}

#[test]
#[should_panic(expected = "lookup_all_sorted requires nondecreasing key prefixes")]
fn sorted_batch_rejects_decreasing_prefixes_within_bucket() {
    let set = sample_set();
    let mut results = set.lookup_all_sorted([*b"abcf", *b"abcd"]);
    assert_eq!(results.next(), Some(false));
    results.next();
}

#[test]
#[should_panic(expected = "lookup_all_sorted requires nondecreasing key prefixes")]
fn sorted_batch_rejects_decreasing_buckets() {
    let set = sample_set();
    let mut results = set.lookup_all_sorted([*b"zzzz", *b"abcd"]);
    assert_eq!(results.next(), Some(true));
    results.next();
}
