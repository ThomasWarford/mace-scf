"""check_train_test_files accepts .xyz files and preprocessed .h5 input."""

import argparse

import pytest

from mace_scf.utils.check_args import check_train_test_files, is_preprocessed_dataset


def file_args(**overrides):
    args = {
        "train_file": None,
        "valid_file": None,
        "test_file": None,
        "test_dir": None,
    }
    args.update(overrides)
    return argparse.Namespace(**args)


@pytest.fixture
def shard_dir(tmp_path):
    """A directory that looks like preprocess_data.py output.

    Only the filenames matter here: the dispatch predicate globs for *.h5 and never opens
    them.
    """
    directory = tmp_path / "train"
    directory.mkdir()
    (directory / "train_0.h5").touch()
    (directory / "train_1.h5").touch()
    return directory


@pytest.fixture
def valid_shard_dir(tmp_path):
    directory = tmp_path / "val"
    directory.mkdir()
    (directory / "val_0.h5").touch()
    return directory


def test_xyz_train_and_valid_accepted(tmp_path):
    train = tmp_path / "train.xyz"
    valid = tmp_path / "valid.xyz"
    train.touch()
    valid.touch()
    check_train_test_files(file_args(train_file=str(train), valid_file=str(valid)))


def test_xyz_train_without_valid_accepted(tmp_path):
    """--valid_fraction is still allowed for .xyz input."""
    train = tmp_path / "train.xyz"
    train.touch()
    check_train_test_files(file_args(train_file=str(train)))


def test_shard_directories_accepted(shard_dir, valid_shard_dir):
    check_train_test_files(
        file_args(train_file=str(shard_dir), valid_file=str(valid_shard_dir))
    )


def test_shard_train_requires_shard_valid(shard_dir):
    with pytest.raises(ValueError, match="valid_file is required when training from preprocessed"):
        check_train_test_files(file_args(train_file=str(shard_dir)))


def test_mixed_xyz_and_shard_inputs_rejected(tmp_path, shard_dir):
    valid = tmp_path / "valid.xyz"
    valid.touch()
    with pytest.raises(ValueError, match="both be .xyz or both be preprocessed"):
        check_train_test_files(
            file_args(train_file=str(shard_dir), valid_file=str(valid))
        )


@pytest.mark.parametrize("suffix", [".txt", ".lmdb", ""])
def test_unsupported_train_file_rejected(tmp_path, suffix):
    train = tmp_path / f"train{suffix}"
    train.touch()
    with pytest.raises(ValueError, match="must be a .xyz file, a .h5 file, or a directory"):
        check_train_test_files(file_args(train_file=str(train)))


def test_missing_train_file_rejected():
    with pytest.raises(ValueError, match="train_file is required"):
        check_train_test_files(file_args())


def test_directory_without_shards_rejected(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "notes.md").touch()
    with pytest.raises(ValueError, match="must be a .xyz file, a .h5 file, or a directory"):
        check_train_test_files(file_args(train_file=str(empty)))


def test_non_xyz_test_file_rejected(tmp_path, shard_dir, valid_shard_dir):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / "test_0.h5").touch()
    with pytest.raises(ValueError, match="Only .xyz test_file"):
        check_train_test_files(
            file_args(
                train_file=str(shard_dir),
                valid_file=str(valid_shard_dir),
                test_file=str(test_dir),
            )
        )


def test_test_dir_still_rejected(shard_dir, valid_shard_dir, tmp_path):
    with pytest.raises(ValueError, match="test_dir"):
        check_train_test_files(
            file_args(
                train_file=str(shard_dir),
                valid_file=str(valid_shard_dir),
                test_dir=str(tmp_path),
            )
        )


def test_single_h5_file_accepted(tmp_path):
    train = tmp_path / "train.h5"
    valid = tmp_path / "valid.h5"
    train.touch()
    valid.touch()
    check_train_test_files(file_args(train_file=str(train), valid_file=str(valid)))


def test_is_preprocessed_dataset(tmp_path, shard_dir):
    xyz = tmp_path / "train.xyz"
    xyz.touch()
    h5 = tmp_path / "train.h5"
    h5.touch()
    empty = tmp_path / "empty"
    empty.mkdir()
    lmdb_dir = tmp_path / "lmdb_shards"
    lmdb_dir.mkdir()
    (lmdb_dir / "data.lmdb").touch()

    assert is_preprocessed_dataset(str(shard_dir))
    assert is_preprocessed_dataset(str(h5))
    assert not is_preprocessed_dataset(str(xyz))
    # An empty directory is "not a dataset" rather than an error, so check_train_test_files
    # can produce its own message.
    assert not is_preprocessed_dataset(str(empty))
    # mace_scf reads HDF5 only; an LMDB directory must not be routed to HDF5Dataset.
    assert not is_preprocessed_dataset(str(lmdb_dir))
