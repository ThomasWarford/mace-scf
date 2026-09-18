# This file loads an xyz dataset and prepares
# new hdf5 file that is ready for training with on-the-fly dataloading

import logging
import ast
import numpy as np
import json
import random
import time
import tqdm
from glob import glob
import h5py
from ase.io import read
import torch
import multiprocessing as mp
import os
import mace
from typing import List, Tuple


from mace import tools, data
from mace.tools.scripts_utils import get_atomic_energies
from mace_scf.utils.check_args import check_and_fix_heads
from mace_scf.utils.load_data import (
    get_atomic_number_table_from_zs,
    log_dataset_summary,
    validate_xyz_collections,
    validate_xyz_paths,
)
from mace_scf.utils.extend_arg_parse import preprocess_extended_arg_parser
from mace.data import save_configurations_as_HDF5, HDF5Dataset
from mace.tools.scripts_utils import (
    get_dataset_from_xyz,
    get_atomic_energies
)
from mace.tools.utils import AtomicNumberTable
from mace.tools import torch_geometric
from mace.modules import compute_statistics


def compute_stats_target(file: str, z_table: AtomicNumberTable, r_max: float, atomic_energies: Tuple, batch_size: int):
    # h5py can transiently fail to open a shard a sibling process just
    # finished writing (stale metadata on a networked filesystem); retry
    # briefly instead of failing the whole preprocessing run over it.
    last_exc = None
    for attempt in range(5):
        try:
            train_dataset = HDF5Dataset(file, z_table=z_table, r_max=r_max)
            break
        except Exception as exc:  # pylint: disable=broad-except
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    else:
        raise last_exc

    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=False,
    )
    
    avg_num_neighbors, mean, std = compute_statistics(train_loader, atomic_energies)
    output = [avg_num_neighbors, float(np.asarray(mean).reshape(-1)[0]), float(np.asarray(std).reshape(-1)[0])]
    return output


def pool_compute_stats(inputs: List): 
    path_to_files, z_table, r_max, atomic_energies, batch_size, num_process = inputs
    pool = mp.Pool(processes=num_process)
    
    re=[pool.apply_async(compute_stats_target, args=(file, z_table, r_max, atomic_energies, batch_size,)) for file in glob(path_to_files+'/*')]
    
    pool.close()
    pool.join()
    results = [r.get() for r in tqdm.tqdm(re)]
    return np.average(results, axis=0)


def split_array(a: np.ndarray, max_size: int):
    drop_last = False
    if len(a) % 2 == 1:
        a = np.append(a, a[-1])
        drop_last = True
    factors = get_prime_factors(len(a))
    max_factor = 1
    for i in range(1, len(factors) + 1):
        for j in range(0, len(factors) - i + 1):
            if np.prod(factors[j : j + i]) <= max_size:
                test = np.prod(factors[j : j + i])
                if test > max_factor:
                    max_factor = test
    return np.array_split(a, max_factor), drop_last


def get_prime_factors(n: int):
    factors = []
    for i in range(2, n + 1):
        while n % i == 0:
            factors.append(i)
            n = n / i
    return factors


def main():
    """
    This script loads an xyz dataset and prepares
    new hdf5 file that is ready for training with on-the-fly dataloading
    """
    args = preprocess_extended_arg_parser().parse_args()
    check_and_fix_heads(args)
    
    # Setup
    tools.set_seeds(args.seed)
    random.seed(args.seed)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    
    try:
        config_type_weights = ast.literal_eval(args.config_type_weights)
        assert isinstance(config_type_weights, dict)
    except Exception as e:  # pylint: disable=W0703
        logging.warning(
            f"Config type weights not specified correctly ({e}), using Default"
        )
        config_type_weights = {"Default": 1.0}
    
    folders = ['train', 'val','test']
    for sub_dir in folders:
        if not os.path.exists(args.h5_prefix+sub_dir):
            os.makedirs(args.h5_prefix+sub_dir)

    args.key_specification = mace.data.KeySpecification()
    head_dict = next(iter(args.heads.values()))
    args.key_specification.update(
        info_keys=head_dict.get("info_keys", {}),
        arrays_keys=head_dict.get("arrays_keys", {}),
    )
    logging.info("Using the key specifications to parse data:")
    logging.info(args.key_specification)

    try:
        config_type_weights = ast.literal_eval(args.config_type_weights)
        assert isinstance(config_type_weights, dict)
    except Exception as e:  # pylint: disable=W0703
        logging.warning(
            f"Config type weights not specified correctly ({e}), using Default"
        )
        config_type_weights = {"Default": 1.0}
    
    # Same validation the .xyz training path runs: once the data is .h5 shards, training
    # can no longer see the pbc/cell/dipole information these checks need.
    validate_xyz_paths(args)
    collections, atomic_energies_dict = get_dataset_from_xyz(
        work_dir=args.work_dir,
        train_path=args.train_file,
        valid_path=args.valid_file,
        valid_fraction=args.valid_fraction,
        config_type_weights=config_type_weights,
        test_path=args.test_file,
        seed=1234,
        key_specification=args.key_specification,

    )
    validate_xyz_collections(collections, args)

    # Atomic number table
    # yapf: disable
    z_table = get_atomic_number_table_from_zs(
        z
        for configs in (collections.train, collections.valid)
        for config in configs
        for z in config.atomic_numbers
    )
    log_dataset_summary(
        z_table, collections.train, collections.valid, collections.tests
    )

    logging.info("Preparing training set")
    if args.shuffle:
        random.shuffle(collections.train)

    # split collections.train into batches and save them to hdf5
    split_train = np.array_split(collections.train,args.num_process)
    drop_last = False
    if len(collections.train) % 2 == 1:
        drop_last = True
    
    # Define Task for Multiprocessiing
    def multi_train_hdf5(process):
        with h5py.File(args.h5_prefix + "train/train_" + str(process)+".h5", "w") as f:
            f.attrs["drop_last"] = drop_last
            save_configurations_as_HDF5(split_train[process], process, f)
      
    processes = []
    for i in range(args.num_process):
        p = mp.Process(target=multi_train_hdf5, args=[i])
        p.start()
        processes.append(p)
        
    for i in processes:
        i.join()


    logging.info("Computing statistics")
    if not atomic_energies_dict:
        atomic_energies_dict = get_atomic_energies(args.E0s, collections.train, z_table)
    atomic_energies: np.ndarray = np.array(
        [atomic_energies_dict[z] for z in z_table.zs]
    )
    logging.info(f"Atomic energies: {atomic_energies.tolist()}")
    _inputs = [args.h5_prefix+'train', z_table, args.r_max, atomic_energies, args.batch_size, args.num_process]
    avg_num_neighbors, mean, std=pool_compute_stats(_inputs)
    logging.info(f"Average number of neighbors: {avg_num_neighbors}")
    logging.info(f"Mean: {mean}")
    logging.info(f"Standard deviation: {std}")

    # Consumers parse these with ast.literal_eval, so the values must be plain Python
    # (get_atomic_number_table_from_zs already keeps z_table.zs int).
    statistics = {
        "atomic_energies": str({int(z): float(e) for z, e in atomic_energies_dict.items()}),
        "avg_num_neighbors": float(avg_num_neighbors),
        "mean": float(mean),
        "std": float(std),
        "atomic_numbers": str(list(z_table.zs)),
        "r_max": args.r_max,
    }
    
    with open(args.h5_prefix + "statistics.json", "w") as f:
        json.dump(statistics, f)
    
    logging.info("Preparing validation set")
    if args.shuffle:
        random.shuffle(collections.valid)
    split_valid = np.array_split(collections.valid, args.num_process) 
    drop_last = False
    if len(collections.valid) % 2 == 1:
        drop_last = True

    def multi_valid_hdf5(process):
        with h5py.File(args.h5_prefix + "val/val_" + str(process)+".h5", "w") as f:
            f.attrs["drop_last"] = drop_last
            save_configurations_as_HDF5(split_valid[process], process, f)
    
    processes = []
    for i in range(args.num_process):
        p = mp.Process(target=multi_valid_hdf5, args=[i])
        p.start()
        processes.append(p)
        
    for i in processes:
        i.join()

    if args.test_file is not None:
        def multi_test_hdf5(process, name):
            with h5py.File(args.h5_prefix + "test/" + name + "_" + str(process) + ".h5", "w") as f:                    
                f.attrs["drop_last"] = drop_last
                save_configurations_as_HDF5(split_test[process], process, f)
            
        logging.info("Preparing test sets")
        for name, subset in collections.tests:
            drop_last = False
            if len(subset) % 2 == 1:
                drop_last = True
            split_test = np.array_split(subset, args.num_process) 

            processes = []
            for i in range(args.num_process):
                p = mp.Process(target=multi_test_hdf5, args=[i, name])
                p.start()
                processes.append(p)

            for i in processes:
                i.join()

if __name__ == "__main__":
    main()