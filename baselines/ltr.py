import sys
import os
import math
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
import numpy as np

from common import UserActivity, ranking_func, split_dataset_by_qids, normalize_features
from ltr_helper import LTRDatasetMaker, write_records

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# torch.mps.manual_seed(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

dev = get_torch_device()

def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    model_size_kb = total_params * 4 / 1024  # Assuming 4 bytes per parameter
    print(f"Model size: {model_size_kb:.2f} KB ({total_params:,} parameters)")

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
    qid_mappings = {(ua.query, ua.issuer) for ua in clicklogs} | {(ua.query, ua.issuer) for ua in activities}
    
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

        test_ds = load_libsvm_dataset_role("test", config.data.path, config.data.slate_length)
        test_dl = DataLoader(test_ds, batch_size=config.data.batch_size, num_workers=config.data.num_workers, shuffle=False)

        # Create a dictionary mapping (query,user) pairs to their first occurrence index
        query_user_to_first_idx = {}
        for idx, activity in enumerate(activities):
            key = (activity.query, activity.issuer)
            if key not in query_user_to_first_idx:
                query_user_to_first_idx[key] = idx
        
        activity_idx = 0
        model.eval()
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
                    # if (query, user) has been processed in a previous iteration, copy the results, and move forward
                    key = (activities[activity_idx].query, activities[activity_idx].issuer)
                    while query_user_to_first_idx[key] < activity_idx:
                        # Get the ranking from the previous activity with same query/user
                        prev_results = activities[query_user_to_first_idx[key]].results
                        curr_results = activities[activity_idx].results
                        
                        # Create mapping from result to position in prev_results
                        result_to_pos = {r: i for i, r in enumerate(prev_results)}
                        
                        # Sort current results based on positions in prev_results
                        activities[activity_idx].results = sorted(curr_results, 
                            key=lambda x: result_to_pos.get(x, len(prev_results)))
                        
                        activity_idx += 1
                        key = (activities[activity_idx].query, activities[activity_idx].issuer)

                    slate_scores = scores[i]
                    slate_indices = indices[i]
                    slate_mask = mask[i]
                    
                    valid_scores = slate_scores[~slate_mask]
                    valid_indices = slate_indices[~slate_mask]
                    
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
        # torch.cuda.empty_cache()
        shutil.rmtree(dataset_path, ignore_errors=True)

    return activities
