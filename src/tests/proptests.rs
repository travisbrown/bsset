// SAFETY: each test owns its temporary files and leaves mapped files unchanged.

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
    let mut builder = ByteStringSet::<KEY_LEN>::builder_with_prefix_len(prefix_len);
    for &key in keys {
        builder.insert(key).expect("in-memory insert cannot fail");
    }
    builder.build().expect("prefix_len is within 3..=KEY_LEN")
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

        let loaded = unsafe { ByteStringSet::<KEY_LEN>::read(&path) }.expect("load data file");
        prop_assert_eq!(loaded.prefix_len(), prefix_len);
        prop_assert_eq!(loaded.len(), set.len());
        for &probe in keys.iter().chain(&probes) {
            prop_assert_eq!(loaded.lookup(probe), set.lookup(probe));
        }
    }

    #[test]
    fn import_matches_direct_inserts(
        file_keys in proptest::collection::vec(key_strategy(), 0..64),
        extra_keys in proptest::collection::vec(key_strategy(), 0..64),
        external in any::<bool>(),
        max_bucket_bytes in 0usize..=128,
        // The imported file's prefix length must be at least the builder's, so draw the
        // pair with `prop_flat_map`: first the file's, then the builder's within it.
        (file_prefix_len, prefix_len) in (OFFSET_TABLE_PREFIX_LEN..=KEY_LEN)
            .prop_flat_map(|file| (Just(file), OFFSET_TABLE_PREFIX_LEN..=file)),
    ) {
        // Write a set of `file_prefix_len`-byte prefixes, import it alongside loose keys,
        // and compare with building from the concatenated key list directly; truncating an
        // imported prefix is the same as truncating the original key.
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("set.bin");
        build_set(&file_keys, file_prefix_len).write(&path).expect("write data file");

        let mut builder = if external {
            ByteStringSet::<KEY_LEN>::external_builder(prefix_len, max_bucket_bytes)
                .expect("create external builder")
        } else {
            ByteStringSet::<KEY_LEN>::builder_with_prefix_len(prefix_len)
        };
        builder.import(&path).expect("import data file");
        for &key in &extra_keys {
            builder.insert(key).expect("insert key");
        }
        let merged = builder.build().expect("valid build");

        let all: Vec<_> = file_keys.iter().chain(&extra_keys).copied().collect();
        let direct = build_set(&all, prefix_len);
        prop_assert_eq!(merged.len(), direct.len());
        for &probe in &all {
            prop_assert!(merged.lookup(probe));
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
        let mut builder =
            ByteStringSet::<KEY_LEN>::external_builder(prefix_len, max_bucket_bytes)
                .expect("create spill file");
        for &key in &keys {
            builder.insert(key).expect("spill key");
        }
        let external = builder.build().expect("valid build");
        prop_assert_eq!(external.len(), in_memory.len());
        for &probe in keys.iter().chain(&probes) {
            prop_assert_eq!(external.lookup(probe), in_memory.lookup(probe));
        }
    }
}
