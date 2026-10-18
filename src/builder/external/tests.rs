use super::*;

struct ReadFailure;

impl Read for ReadFailure {
    fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
        Err(std::io::Error::other("injected read failure"))
    }
}

#[test]
fn failed_import_cannot_build_a_partial_key() {
    let mut spill = SpillState::<4>::new(4).expect("create spill file");
    let mut reader = b"abcdx".as_slice().chain(ReadFailure);
    assert!(spill.import(&mut reader, 2, 4).is_err());
    assert!(spill.insert(b"wxyz").is_err());
    assert!(spill.import(&mut b"wxyz".as_slice(), 1, 4).is_err());
    assert!(spill.build(4).is_err());
}

#[test]
fn truncated_import_invalidates_full_and_padded_builders() {
    for prefix_len in [3, 4] {
        let mut spill = SpillState::<4>::new(4).expect("create spill file");
        let error = spill
            .import(&mut b"abcdx".as_slice(), 2, prefix_len)
            .expect_err("input is shorter than two entries");
        assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof);
        assert!(spill.build(3).is_err());
    }
}

#[test]
fn import_stops_at_the_validated_length() {
    let mut spill = SpillState::<4>::new(4).expect("create spill file");
    let mut reader = b"abcdextra".as_slice();
    spill.import(&mut reader, 1, 4).expect("import one key");
    assert_eq!(reader, b"extra");
    spill.insert(b"wxyz").expect("append another key");
    let set = spill.build(4).expect("build imported keys");
    assert_eq!(set.len(), 2);
    assert!(set.lookup(*b"abcd"));
    assert!(set.lookup(*b"wxyz"));
    assert!(!set.lookup(*b"extr"));
}

#[test]
fn common_prefix_bytes_do_not_require_separate_partition_passes() {
    let mut file = tempfile::tempfile().expect("create input");
    let mut key = [0; 4096];
    // Separate first-byte buckets with the same long suffix can both skip to the end.
    for first in [0, 1, 0, 1] {
        key[0] = first;
        file.write_all(&key).expect("write key");
    }
    file.rewind().expect("rewind input");
    let mut pass = ExternalBuild::<4096> {
        prefix_len: 4096,
        max_bucket_bytes: 4096,
        partition_dir: tempfile::tempdir().expect("create partition directory"),
        next_file_id: 0,
        out: BufWriter::new(tempfile::tempfile().expect("create output")),
    };
    pass.run(file, 4 * 4096).expect("partition and emit");
    assert_eq!(pass.next_file_id, 2, "one partition file per first byte");
    assert_eq!(
        std::fs::read_dir(pass.partition_dir.path())
            .expect("list partitions")
            .count(),
        0
    );
    let mut out = pass.out.into_inner().expect("flush output");
    out.rewind().expect("rewind output");
    let mut actual = Vec::new();
    out.read_to_end(&mut actual).expect("read output");
    let mut expected = vec![0; 2 * 4096];
    expected[4096] = 1;
    assert_eq!(actual, expected);
}

#[test]
fn deep_partitioning_uses_a_bounded_call_stack() {
    // The first differing byte advances by one in every pass. A small stack catches a return
    // to recursive traversal without requiring enormous keys or millions of entries.
    std::thread::Builder::new()
        .stack_size(128 * 1024)
        .spawn(|| {
            let mut builder =
                ByteStringSet::<256>::external_builder(256, 256).expect("create builder");
            builder.insert([0; 256]).expect("insert zero key");
            for index in 0..256 {
                let mut key = [0; 256];
                key[index] = 1;
                builder.insert(key).expect("insert key");
            }
            let set = builder.build().expect("build deep partitions");
            assert_eq!(set.len(), 257);
            assert!(set.lookup([0; 256]));
            for index in 0..256 {
                let mut key = [0; 256];
                key[index] = 1;
                assert!(set.lookup(key));
            }
        })
        .expect("spawn small-stack worker")
        .join()
        .expect("finish deep build");
}

#[test]
fn bytes_shared_within_one_partition_are_skipped() {
    // Byte 1 differs between the two first-byte partitions but is shared within each, so
    // neither partition should be copied whole on byte 1 before splitting on byte 3.
    let mut file = tempfile::tempfile().expect("create input");
    for key in [b"aXa1", b"aXa2", b"bYb1", b"bYb2"] {
        file.write_all(key).expect("write key");
    }
    file.rewind().expect("rewind input");
    let mut pass = ExternalBuild::<4> {
        prefix_len: 4,
        max_bucket_bytes: 4,
        partition_dir: tempfile::tempdir().expect("create partition directory"),
        next_file_id: 0,
        out: BufWriter::new(tempfile::tempfile().expect("create output")),
    };
    pass.run(file, 16).expect("partition and emit");
    assert_eq!(
        pass.next_file_id, 6,
        "two first-byte partitions, each split once"
    );
    let mut out = pass.out.into_inner().expect("flush output");
    out.rewind().expect("rewind output");
    let mut actual = Vec::new();
    out.read_to_end(&mut actual).expect("read output");
    assert_eq!(actual, b"aXa1aXa2bYb1bYb2");
}
