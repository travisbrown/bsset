#![expect(unsafe_code, reason = "tests load sets with `ByteStringSet::read`")]

// SAFETY: each test owns its temporary files and leaves mapped files unchanged.

mod lookup;
mod proptests;

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
    builder.build().expect("valid build")
}

/// Build a set with the external-sort builder from `keys`.
fn external_set<const N: usize>(
    max_bucket_bytes: usize,
    keys: &[[u8; N]],
    prefix_len: usize,
) -> ByteStringSet<N> {
    let mut builder = ByteStringSet::<N>::external_builder(prefix_len, max_bucket_bytes)
        .expect("create spill file");
    for &key in keys {
        builder.insert(key).expect("spill key");
    }
    builder.build().expect("valid build")
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
    let mut builder = ByteStringSet::<8>::builder_with_prefix_len(4);
    builder.insert(*b"abcd1234").expect("insert key");
    builder.insert(*b"wxyz0000").expect("insert key");
    let set = builder.build().expect("valid build");
    // Only the first 4 bytes are kept, so any suffix matches.
    assert!(set.lookup(*b"abcdXXXX"));
    assert!(set.lookup(*b"wxyz9999"));
    assert!(!set.lookup(*b"abzz1234"));
}

#[test]
fn duplicates_and_shared_prefixes_are_deduped() {
    let mut builder = ByteStringSet::<8>::builder_with_prefix_len(4);
    builder.insert(*b"abcd1234").expect("insert key");
    builder.insert(*b"abcd1234").expect("insert key");
    builder.insert(*b"abcd9999").expect("insert key");
    let set = builder.build().expect("valid build");
    assert_eq!(set.len(), 1);
}

#[test]
fn empty_set_misses_everything() {
    let set = ByteStringSet::<4>::builder().build().expect("valid build");
    assert!(set.is_empty());
    assert!(!set.lookup(*b"abcd"));
}

#[test]
fn prefix_len_bounds_are_enforced() {
    assert!(matches!(
        ByteStringSet::<4>::builder_with_prefix_len(2).build(),
        Err(Error::InvalidPrefixLen {
            prefix_len: 2,
            key_len: 4
        })
    ));
    assert!(matches!(
        ByteStringSet::<4>::builder_with_prefix_len(5).build(),
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

    let loaded = unsafe { ByteStringSet::<4>::read(&path) }.expect("load data file");
    assert_eq!(loaded.len(), 2);
    assert!(loaded.lookup(*b"abcd"));
    assert!(loaded.lookup(*b"zzzz"));
    assert!(!loaded.lookup(*b"abce"));
}

#[test]
fn import_merges_file_into_builder() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("set.bin");
    sample_set().write(&path).expect("write data file");

    let mut builder = ByteStringSet::<4>::builder();
    builder.insert(*b"wxyz").expect("insert key");
    builder.import(&path).expect("import data file");
    let set = builder.build().expect("valid build");
    assert_eq!(set.len(), 5);
    assert!(set.lookup(*b"wxyz"));
    assert!(set.lookup(*b"abcd"));
    assert!(set.lookup(*b"zzzz"));
    assert!(!set.lookup(*b"abcf"));
}

#[test]
fn import_merges_file_into_external_builder() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("set.bin");
    sample_set().write(&path).expect("write data file");

    // A one-key bucket budget forces the imported keys through the partitioning path.
    let mut builder = ByteStringSet::<4>::external_builder(4, 4).expect("create spill file");
    builder.insert(*b"abcd").expect("spill key");
    builder.insert(*b"wxyz").expect("spill key");
    builder.import(&path).expect("import data file");
    let set = builder.build().expect("valid build");
    // "abcd" was inserted and imported; it must still collapse to one entry.
    assert_eq!(set.len(), 5);
    assert!(set.lookup(*b"wxyz"));
    assert!(set.lookup(*b"abcd"));
    assert!(set.lookup(*b"zzzz"));
}

/// Write a data file of 4-byte prefixes (from an 8-byte-key set) and return its path.
fn write_prefix_file(dir: &Path) -> std::path::PathBuf {
    let mut builder = ByteStringSet::<8>::builder_with_prefix_len(4);
    builder.insert(*b"abcd1234").expect("insert key");
    let set = builder.build().expect("valid build");
    let path = dir.join("prefixes.bin");
    set.write(&path).expect("write data file");
    path
}

#[test]
fn import_pads_shorter_prefixes() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = write_prefix_file(dir.path());

    let mut builder = ByteStringSet::<8>::builder_with_prefix_len(4);
    builder.import(&path).expect("import data file");
    builder.insert(*b"wxyz5678").expect("insert key");
    let set = builder.build().expect("valid build");
    assert_eq!(set.len(), 2);
    assert!(set.lookup(*b"abcdXXXX"));
    assert!(set.lookup(*b"wxyz9999"));
    assert!(!set.lookup(*b"aaaaXXXX"));
}

#[test]
fn import_pads_into_external_builder() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = write_prefix_file(dir.path());

    // An 8-byte bucket budget forces the padded keys through the partitioning path.
    let mut builder = ByteStringSet::<8>::external_builder(3, 8).expect("create spill file");
    builder.import(&path).expect("import data file");
    builder.insert(*b"wxyz5678").expect("spill key");
    let set = builder.build().expect("valid build");
    assert_eq!(set.len(), 2);
    assert!(set.lookup(*b"abcXXXXX"));
    assert!(set.lookup(*b"wxyXXXXX"));
}

#[test]
fn import_rejects_shorter_prefix_file() {
    // The file stores 4-byte prefixes, which carry no information past byte 4, so a
    // builder targeting longer prefixes must refuse them.
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = write_prefix_file(dir.path());

    let mut builder = ByteStringSet::<8>::builder();
    assert!(matches!(
        builder.import(&path),
        Err(Error::PrefixLenExceedsImport {
            prefix_len: 8,
            imported_prefix_len: 4
        })
    ));
}

#[test]
fn import_rejects_invalid_files() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("bad.bin");
    let mut builder = ByteStringSet::<4>::builder();

    std::fs::write(&path, b"junk").expect("write file");
    assert!(matches!(builder.import(&path), Err(Error::BadHeader)));

    std::fs::write(&path, raw_file(4, b"abcde")).expect("write file");
    assert!(matches!(
        builder.import(&path),
        Err(Error::Misaligned {
            data_len: 5,
            prefix_len: 4
        })
    ));

    std::fs::write(&path, raw_file(5, b"abcde")).expect("write file");
    assert!(matches!(
        builder.import(&path),
        Err(Error::InvalidPrefixLen {
            prefix_len: 5,
            key_len: 4
        })
    ));

    // The failed imports must not have contaminated the builder.
    let set = builder.build().expect("valid build");
    assert!(set.is_empty());
}

#[test]
fn file_roundtrip_preserves_membership() {
    let set = sample_set();
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("set.bin");
    set.write(&path).expect("write data file");

    let loaded = unsafe { ByteStringSet::<4>::read(&path) }.expect("load data file");
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
        unsafe { ByteStringSet::<4>::read(&path) },
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
        unsafe { ByteStringSet::<4>::read(&path) },
        Err(Error::Unsorted { index: 1 })
    ));
}

#[test]
fn read_rejects_missing_or_truncated_header() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("garbage.bin");
    std::fs::write(&path, []).expect("write empty file");
    assert!(matches!(
        unsafe { ByteStringSet::<4>::read(&path) },
        Err(Error::BadHeader)
    ));

    std::fs::write(&path, b"not a bsset file").expect("write file");
    assert!(matches!(
        unsafe { ByteStringSet::<4>::read(&path) },
        Err(Error::BadHeader)
    ));

    std::fs::write(&path, b"BSST").expect("write file");
    assert!(matches!(
        unsafe { ByteStringSet::<4>::read(&path) },
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
        unsafe { ByteStringSet::<4>::read(&path) },
        Err(Error::UnsupportedVersion { version: 2 })
    ));
}

#[test]
fn read_rejects_prefix_exceeding_key_len() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("wide.bin");
    std::fs::write(&path, raw_file(5, b"abcde")).expect("write file");
    assert!(matches!(
        unsafe { ByteStringSet::<4>::read(&path) },
        Err(Error::InvalidPrefixLen {
            prefix_len: 5,
            key_len: 4
        })
    ));
}

#[cfg(unix)]
#[test]
fn write_replaces_a_mapped_file_without_changing_live_sets() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("set.bin");
    let original = sample_set();
    original.write(&path).expect("write original set");
    let loaded = unsafe { ByteStringSet::<4>::read(&path) }.expect("map original set");

    loaded.write(&path).expect("write over own mapping");
    let mut builder = ByteStringSet::<4>::builder();
    builder.insert(*b"new!").expect("insert replacement key");
    builder
        .build()
        .expect("build replacement")
        .write(&path)
        .expect("replace mapped file");

    assert_eq!(loaded.len(), original.len());
    assert!(loaded.lookup(*b"abcd"));
    assert!(!loaded.lookup(*b"new!"));
    let replacement = unsafe { ByteStringSet::<4>::read(&path) }.expect("map replacement set");
    assert_eq!(replacement.len(), 1);
    assert!(replacement.lookup(*b"new!"));
    assert!(!replacement.lookup(*b"abcd"));
}

#[test]
fn failed_write_leaves_destination_and_directory_intact() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let destination = dir.path().join("existing-directory");
    std::fs::create_dir(&destination).expect("create destination directory");
    let marker = destination.join("marker");
    std::fs::write(&marker, b"keep me").expect("write marker");

    assert!(sample_set().write(&destination).is_err());
    assert_eq!(std::fs::read(marker).expect("read marker"), b"keep me");
    assert_eq!(
        std::fs::read_dir(dir.path()).expect("list parent").count(),
        1
    );
}

#[cfg(unix)]
#[test]
fn write_replaces_symlinks_and_preserves_hard_links() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let original = dir.path().join("original");
    let hard_link = dir.path().join("hard-link");
    let symlink = dir.path().join("symlink");
    std::fs::write(&original, b"keep me").expect("write original");
    std::fs::hard_link(&original, &hard_link).expect("create hard link");
    std::os::unix::fs::symlink(&original, &symlink).expect("create symlink");

    let set = sample_set();
    set.write(&hard_link).expect("replace hard link");
    set.write(&symlink).expect("replace symlink");
    assert_eq!(std::fs::read(&original).expect("read original"), b"keep me");
    assert!(
        !std::fs::symlink_metadata(&symlink)
            .expect("stat symlink path")
            .is_symlink()
    );
    assert_eq!(
        std::fs::read(hard_link).expect("read set"),
        std::fs::read(symlink).expect("read set")
    );
}

#[cfg(unix)]
#[test]
fn write_uses_default_file_permissions() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("set.bin");
    let reference = dir.path().join("reference");
    sample_set().write(&path).expect("write set");
    std::fs::File::create(&reference).expect("create reference file");

    let mode = |path| {
        std::fs::metadata(path)
            .expect("stat file")
            .permissions()
            .mode()
            & 0o777
    };
    assert_eq!(mode(&path), mode(&reference));
}

#[test]
fn duplicate_keys_do_not_inflate_the_finished_buffer() {
    let mut builder = ByteStringSet::<64>::builder_with_prefix_len(4);
    for suffix in 0..10_000_u64 {
        let mut key = [0_u8; 64];
        key[..4].copy_from_slice(b"same");
        key[4..12].copy_from_slice(&suffix.to_be_bytes());
        builder.insert(key).expect("insert key");
    }
    let set = builder.build().expect("build set");
    assert_eq!(set.len(), 1);
    let Storage::Owned(data) = &set.data else {
        panic!("in-memory build must own its data");
    };
    assert_eq!(data, b"same");
    // Sizing the buffer before deduplication would reserve a stride for every inserted key.
    assert!(data.capacity() < 10_000 * 4);
}

#[test]
fn offset_table_handles_boundary_buckets_and_gaps() {
    let data = [
        [0, 0, 0, 1],
        [0, 0, 0, 2],
        [0, 0, 2, 0],
        [255, 255, 255, 0],
        [255, 255, 255, 255],
    ];
    let offsets = build_offsets(data.as_flattened(), 4).expect("build offsets");
    assert_eq!(&offsets[..4], &[0, 2, 2, 3]);
    assert!(
        offsets[3..OFFSET_TABLE_LEN - 1]
            .iter()
            .all(|offset| *offset == 3)
    );
    assert_eq!(offsets[OFFSET_TABLE_LEN - 1], 5);

    let set = external_set(4, &data, 4);
    for key in data {
        assert!(set.lookup(key));
    }
    for key in [
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [255, 255, 254, 0],
        [255, 255, 255, 1],
    ] {
        assert!(!set.lookup(key));
    }
}

#[test]
fn read_rejects_duplicate_prefixes() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("duplicate.bin");
    std::fs::write(&path, raw_file(3, b"abcabc")).expect("write file");
    assert!(matches!(
        unsafe { ByteStringSet::<3>::read(&path) },
        Err(Error::Unsorted { index: 1 })
    ));
}

#[test]
fn writing_a_mapped_path_preserves_its_data_even_if_replacement_is_denied() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("set.bin");
    let original = sample_set();
    original.write(&path).expect("write set");
    let expected = std::fs::read(&path).expect("read original bytes");
    let loaded = unsafe { ByteStringSet::<4>::read(&path) }.expect("map set");

    // Windows and some filesystems may reject replacement while the old file is mapped.
    // Either outcome must preserve the map and leave a complete file at the destination.
    let result = loaded.write(&path);
    if !cfg!(windows) {
        result.expect("replace mapped file");
    }
    assert_eq!(std::fs::read(&path).expect("read saved bytes"), expected);
    assert_eq!(loaded.len(), original.len());
    assert!(loaded.lookup(*b"abcd"));
    assert!(loaded.lookup(*b"zzzz"));
    drop(loaded);
    original.write(&path).expect("replace unmapped file");
}
