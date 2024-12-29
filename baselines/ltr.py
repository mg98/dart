import sys
import os
import time
from urllib.parse import urlparse

sys.path.append('./allRank')

import allrank.models.losses as losses
import numpy as np
from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset, create_data_loaders
from allrank.models.model import make_model
from allrank.models.model_utils import get_torch_device, CustomDataParallel
from allrank.training.train_utils import fit
from allrank.utils.command_executor import execute_command
from allrank.utils.experiments import dump_experiment_result, assert_expected_metrics
from allrank.utils.file_utils import create_output_dirs, PathsContainer, copy_local_to_gs
from allrank.utils.ltr_logging import init_logger
from allrank.utils.python_utils import dummy_context_mgr

import torch
from torch.utils.data import DataLoader
from common import UserActivity
from argparse import Namespace
from attr import asdict
from functools import partial
from pprint import pformat
from torch import optim

def shard_dataset(dataset, shard_id, num_shards):
    """
    Splits the dataset into shards.
    
    Args:
        dataset: The original dataset.
        shard_id: The ID of the shard.
        num_shards: The total number of shards.
    
    Returns:
        A subset of the dataset corresponding to the shard.
    """
    # Calculate shard size
    total_size = len(dataset)
    shard_size = total_size // num_shards

    # Determine the start and end indices of the current shard
    start_idx = shard_id * shard_size
    end_idx = (shard_id + 1) * shard_size if shard_id < num_shards - 1 else total_size

    # Return the subset of the dataset for the current shard
    return torch.utils.data.Subset(dataset, range(start_idx, end_idx)).dataset

def ltr_rank(user_activities: list[UserActivity], shard_id=0, num_shards=3):
    """
    Implementing Learning to Rank using the allRank library with dataset sharding.
    Args:
        user_activities (list[UserActivity]): The list of user activities.
        shard_id (int): The ID of the current shard (for sharding purposes).
        num_shards (int): The total number of shards.
    """

    args = Namespace(
        job_dir="./",
        run_id=str(int(time.time())),
        config_file_name="./allRank/scripts/local_config_click_model.json"
    )

    paths = PathsContainer.from_args(args.job_dir, args.run_id, args.config_file_name)

    create_output_dirs(paths.output_dir)

    logger = init_logger(paths.output_dir)
    logger.info(f"created paths container {paths}")

    # read config
    config = Config.from_json(paths.config_path)
    logger.info("Config:\n {}".format(pformat(vars(config), width=1)))

    output_config_path = os.path.join(paths.output_dir, "used_config.json")
    execute_command("cp {} {}".format(paths.config_path, output_config_path))

    # train_ds, val_ds
    train_ds, val_ds = load_libsvm_dataset(
        input_path=config.data.path,
        slate_length=config.data.slate_length,
        validation_ds_role=config.data.validation_ds_role,
    )

    # Shard the datasets
    train_ds = shard_dataset(train_ds, shard_id, num_shards)
    val_ds = shard_dataset(val_ds, shard_id, num_shards)

    n_features = train_ds.shape[-1]
    print("n features", n_features)
    assert n_features == val_ds.shape[-1], "Last dimensions of train_ds and val_ds do not match!"

    # train_dl, val_dl
    train_dl, val_dl = create_data_loaders(
        train_ds, val_ds, num_workers=config.data.num_workers, batch_size=config.data.batch_size)

    # gpu support
    dev = get_torch_device()
    logger.info("Model training will execute on {}".format(dev.type))

    # instantiate model
    model = make_model(n_features=n_features, **asdict(config.model, recurse=False))
    if torch.cuda.device_count() > 1:
        model = CustomDataParallel(model)
        logger.info("Model training will be distributed to {} GPUs.".format(torch.cuda.device_count()))
    model.to(dev)

    # load optimizer, loss and LR scheduler
    optimizer = getattr(optim, config.optimizer.name)(params=model.parameters(), **config.optimizer.args)
    loss_func = partial(getattr(losses, config.loss.name), **config.loss.args)
    if config.lr_scheduler.name:
        scheduler = getattr(optim.lr_scheduler, config.lr_scheduler.name)(optimizer, **config.lr_scheduler.args)
    else:
        scheduler = None

    with torch.autograd.detect_anomaly() if config.detect_anomaly else dummy_context_mgr():  # type: ignore
        # run training
        result = fit(
            model=model,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=val_dl,
            config=config,
            device=dev,
            output_dir=paths.output_dir,
            tensorboard_output_path=paths.tensorboard_output_path,
            **asdict(config.training)
        )

    dump_experiment_result(args, config, paths.output_dir, result)

    if urlparse(args.job_dir).scheme == "gs":
        copy_local_to_gs(paths.local_base_output_path, args.job_dir)

    assert_expected_metrics(result, config.expected_metrics)
    
    return None
