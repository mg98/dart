import sys
import os
import time
import shutil
import uuid

import allrank.models.losses as losses
from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset, load_libsvm_dataset_role, create_data_loaders
from allrank.models.model import make_model
from allrank.models.model_utils import get_torch_device, load_state_dict_from_file
from allrank.training.train_utils import fit
from allrank.utils.file_utils import create_output_dirs, PathsContainer
from allrank.utils.python_utils import dummy_context_mgr
from allrank.inference.inference_utils import rank_slates, metrics_on_clicked_slates
from allrank.click_models.click_utils import click_on_slates
from allrank.utils.config_utils import instantiate_from_recursive_name_args
from allrank.models.metrics import ndcg, dcg

import torch
from torch.utils.data import DataLoader
from argparse import Namespace
from attr import asdict
from functools import partial
from torch import optim
from copy import deepcopy
import contextlib

from common import UserActivity, ranking_func, split_dataset_by_qids, normalize_features
from ltr_helper import LTRDatasetMaker, write_records

dev = get_torch_device()


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

def train_ltr_model(config):
    train_ds, val_ds = load_libsvm_dataset(
        input_path=config.data.path,
        slate_length=config.data.slate_length,
        validation_ds_role=config.data.validation_ds_role,
    )
    train_dl, val_dl = create_data_loaders(
        train_ds, val_ds, num_workers=config.data.num_workers, batch_size=config.data.batch_size)

    model = make_model(n_features=train_ds.shape[-1], **asdict(config.model, recurse=False))
    model.to(dev)

    optimizer = getattr(optim, config.optimizer.name)(params=model.parameters(), **config.optimizer.args)
    loss_func = partial(getattr(losses, config.loss.name), **config.loss.args)
    if config.lr_scheduler.name:
        scheduler = getattr(optim.lr_scheduler, config.lr_scheduler.name)(optimizer, **config.lr_scheduler.args)
    else:
        scheduler = None
    
    with torch.autograd.detect_anomaly() if config.detect_anomaly else dummy_context_mgr():  # type: ignore
        fit(
            model=model,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=val_dl,
            config=config,
            device=dev,
            **asdict(config.training)
        )

    return model

def test_ltr_model(model, paths, config):
    datasets = {"test": load_libsvm_dataset_role("test", config.data.path, config.data.slate_length)}

    click_model = instantiate_from_recursive_name_args(name_args=config.click_model)
    with contextlib.redirect_stdout(None):
        ranked_slates = rank_slates(datasets, model, config)
    clicked_slates = click_on_slates(ranked_slates["test"], click_model, include_empty=False)

    Xs, ys = clicked_slates
    
    metrics = {
        "ndcg_5": [],
        "ndcg_10": [],
        "ndcg_30": [],
        "ndcg_60": []
    }
    for X, y in zip(Xs, ys):
        ndcg_scores = ndcg(
            torch.arange(start=len(y), end=0, step=-1, dtype=torch.float32)[None, :],
            torch.tensor(y)[None, :],
            ats=[5, 10, 30, 60]
        )
        
        metrics["ndcg_5"].append(ndcg_scores[0][0].item())
        metrics["ndcg_10"].append(ndcg_scores[0][1].item())
        metrics["ndcg_30"].append(ndcg_scores[0][2].item())
        metrics["ndcg_60"].append(ndcg_scores[0][3].item())

    # calculate metrics
    metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
    print(metrics)
    
    # metered_slates = {role: metrics_on_clicked_slates(slates) for role, slates in clicked_slates.items()}  # type: ignore
    # for role, metrics in metered_slates.items():
    #     import pandas as pd
    #     metrics_df = pd.DataFrame(metrics)
    #     metrics_df.to_csv(os.path.join(paths.output_dir, f"{role}_metrics.csv"), index=False)
    #     print(os.path.join(paths.output_dir, f"{role}_metrics_mean.csv"))
    #     pd.DataFrame(metrics_df.mean()).T.to_csv(os.path.join(paths.output_dir, f"{role}_metrics_mean.csv"), index=False)


@ranking_func
def ltr_rank(clicklogs: list[UserActivity], activities: list[UserActivity]):
    """
    Implementing Learning to Rank using the allRank library with dataset sharding.
    Args:
        user_activities (list[UserActivity]): The list of user activities.
        shard_id (int): The ID of the current shard (for sharding purposes).
        num_shards (int): The total number of shards.
    """
    config = Config.from_json("./allRank_config.json")

    dataset_path = f'.tmp/{uuid.uuid4().hex}/'
    print(dataset_path)
    qid_mappings = {ua.query for ua in clicklogs} | {ua.query for ua in activities}
    
    # create train.txt and vali.txt
    ltrdm_clicklogs = LTRDatasetMaker(clicklogs)
    ltrdm_clicklogs.qid_mappings = qid_mappings
    records = ltrdm_clicklogs.compile_records()
    train_records, vali_records, _ = split_dataset_by_qids(records, train_ratio=0.8, val_ratio=0.2)
    del records

    # create test.txt
    ltrdm_activities = LTRDatasetMaker(activities)
    ltrdm_activities.qid_mappings = qid_mappings
    test_records = ltrdm_activities.compile_records()
    
    write_records(dataset_path, {
        "train": train_records,
        "vali": vali_records,
        "test": test_records
    })
    normalize_features(dataset_path)
    
    config.data.path = os.path.join(dataset_path, "_normalized")

    try:
        model = train_ltr_model(deepcopy(config))
        # test_ltr_model(model, paths, deepcopy(config))

        test_ds = load_libsvm_dataset_role("test", config.data.path, config.data.slate_length)
        
        test_dl = DataLoader(test_ds, batch_size=config.data.batch_size, num_workers=config.data.num_workers, shuffle=False)

        model.eval()
        activity_idx = 0

        # Create a dictionary mapping queries to their first occurrence index
        query_to_first_idx = {}
        for idx, activity in enumerate(activities):
            if activity.query not in query_to_first_idx:
                query_to_first_idx[activity.query] = idx
        
        with torch.no_grad():
            for xb, yb, indices in test_dl:

                X = xb.type(torch.float32).to(device=dev)
                y_true = yb.to(device=dev)
                indices = indices.to(device=dev)

                input_indices = torch.ones_like(y_true).type(torch.long)
                mask = (y_true == losses.PADDED_Y_VALUE)
                scores = model.score(X, mask, input_indices)

                # Iterate over each query in the batch
                for i in range(scores.size(0)):
                    while query_to_first_idx[activities[activity_idx].query] < activity_idx:
                        activities[activity_idx].results = activities[
                                query_to_first_idx[activities[activity_idx].query]
                            ].results
                        activity_idx += 1

                    slate_scores = scores[i]
                    slate_indices = indices[i]
                    slate_mask = mask[i]
                    
                    valid_scores = slate_scores[~slate_mask]
                    valid_indices = slate_indices[~slate_mask]
                    
                    # if valid_scores.numel() == 0:
                    #     print(f"Query has no valid documents.")
                    #     continue
                    
                    # Compute the rankings
                    _, sorted_idx = torch.sort(valid_scores, descending=True)
                    sorted_original_indices = valid_indices[sorted_idx]

                    sorted_indices = sorted_original_indices.cpu().tolist()

                    activities[activity_idx].results = [
                        activities[activity_idx].results[i] for i in sorted_indices
                    ]
                    activity_idx += 1

    except Exception as e:
        raise e
    finally:
        pass
        # shutil.rmtree(dataset_path, ignore_errors=True)

    return activities
