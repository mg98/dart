import sys
import os
import math
import shutil
import uuid
import pickle
from dataclasses import dataclass

import allrank.models.losses as losses
from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset, load_libsvm_dataset_role, create_data_loaders
from allrank.models.model import make_model
from allrank.models.model_utils import get_torch_device
from allrank.training.train_utils import fit
from allrank.utils.python_utils import dummy_context_mgr

import torch
from torch.utils.data import DataLoader
from argparse import Namespace
from attr import asdict
from functools import partial
from torch import optim
from copy import deepcopy
import numpy as np

from utils.common import UserActivity, ClickThroughRecord, ranking_func, split_dataset_by_qids, normalize_features, QueryDocumentRelationVector, Corpus
from utils.ltr_helper import LTRDatasetMaker, write_records, qid_key

from baselines.panache import compute_hit_counts
from baselines.maay import MAAY
from baselines.dinx import compute_click_counts
from baselines.grank import precompute_grank_score_fn

torch.manual_seed(42)
torch.cuda.manual_seed(42)
if hasattr(torch, 'mps'): 
    torch.mps.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dev = get_torch_device()

@dataclass
class PrecomputedData:
    hit_counts: dict[str, int]
    click_counts: dict[str, int]
    maay: dict[str, float]
    grank: dict[str, float]
    model: torch.nn.Module
    train_records: list[ClickThroughRecord]
    vali_records: list[ClickThroughRecord]

def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    model_size_kb = total_params * 4 / 1024  # Assuming 4 bytes per parameter
    print(f"Model size: {model_size_kb:.2f} KB ({total_params:,} parameters)")

def create_trained_model(config, training=True):
    """
    Creates and optionally trains a Learning to Rank model based on the provided configuration.
    
    Args:
        config (Config): Configuration object containing model architecture and training parameters
        training (bool): If True, loads training data and trains the model. If False, just creates model.
        
    Returns:
        model: The created (and optionally trained) model
    """
    n_features = len(QueryDocumentRelationVector().features)
    model = make_model(n_features=n_features, **asdict(config.model, recurse=False))
    print_model_size(model)
    model.to(dev)

    optimizer = getattr(optim, config.optimizer.name)(params=model.parameters(), **config.optimizer.args)
    loss_func = partial(getattr(losses, config.loss.name), **config.loss.args)
    if config.lr_scheduler.name:
        scheduler = getattr(optim.lr_scheduler, config.lr_scheduler.name)(optimizer, **config.lr_scheduler.args)
    else:
        scheduler = None

    if training:
        train_ds, val_ds = load_libsvm_dataset(
            input_path=config.data.path,
            slate_length=config.data.slate_length,
            validation_ds_role=config.data.validation_ds_role,
        )
        train_dl, val_dl = create_data_loaders(
            train_ds, val_ds, num_workers=config.data.num_workers, batch_size=config.data.batch_size)
        
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
def ltr_rank(clicklogs: list[UserActivity], activities: list[UserActivity], config: Config, precompute: bool = False, prec_data = None):

    if config.data.path == '':
        dataset_path = f'.tmp/{uuid.uuid4().hex}/'
        qid_mappings = {qid_key(ua) for ua in clicklogs} | {qid_key(ua) for ua in activities}
        
        unique_documents = {doc.infohash: doc for ua in clicklogs + activities for doc in ua.results}.values()
        corpus = {doc.infohash: doc.torrent_info.title.lower() for doc in unique_documents}
        corpus = Corpus(corpus)

        # create train.txt and vali.txt
        if prec_data is None:
            ltrdm_clicklogs = LTRDatasetMaker(clicklogs)
            ltrdm_clicklogs.corpus = corpus
            ltrdm_clicklogs.qid_mappings = qid_mappings
            records = ltrdm_clicklogs.compile_records()
            train_records, vali_records, _ = split_dataset_by_qids(records, train_ratio=0.8, val_ratio=0.2)
            del records
        else:
            # Create dummy records with at least one valid entry
            train_records = prec_data.train_records
            vali_records = prec_data.vali_records

        # create test.txt
        ltrdm_activities = LTRDatasetMaker(activities)
        ltrdm_activities.corpus = corpus
        ltrdm_activities.qid_mappings = qid_mappings
        ltrdm_activities.hit_counts = compute_hit_counts(clicklogs) if prec_data is None else prec_data.hit_counts
        ltrdm_activities.click_counts = compute_click_counts(clicklogs) if prec_data is None else prec_data.click_counts
        test_records = ltrdm_activities.compile_records()
        
        training = train_records and vali_records and test_records

        if training:
            write_records(dataset_path, {
                "train": train_records,
                "vali": vali_records,
                "test": test_records
            })
            normalize_features(dataset_path)
            config.data.path = os.path.join(dataset_path, "_normalized")
    else:
        # skip compiling ltr dataset; override test activities
        with open('tribler_data/test_activities.pkl', 'rb') as f:
            activities = pickle.load(f)

    try:
        model = create_trained_model(deepcopy(config), training=True) if prec_data is None else prec_data.model
        torch.save(model.state_dict(), 'dart_model.pt')

        test_ds = load_libsvm_dataset_role("test", config.data.path, config.data.slate_length)
        test_dl = DataLoader(test_ds, batch_size=config.data.batch_size, num_workers=config.data.num_workers)
        
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
                    slate_scores = scores[i]
                    slate_indices = indices[i]
                    slate_mask = mask[i]
                    
                    valid_scores = slate_scores[~slate_mask]
                    valid_indices = slate_indices[~slate_mask]
                    
                    # Compute the rankings
                    _, sorted_idx = torch.sort(valid_scores, descending=True)
                    sorted_original_indices = valid_indices[sorted_idx]

                    sorted_indices = sorted_original_indices.cpu().tolist()

                    prev_results = activities[activity_idx].results
                    activities[activity_idx].results = [
                        prev_results[i] for i in sorted_indices
                    ]
                    
                    activity_idx += 1

    except Exception as e:
        raise e
    finally:
        if config.data.path.startswith('.tmp'):
            shutil.rmtree(config.data.path, ignore_errors=True)

    if precompute:
        return activities, PrecomputedData(
            hit_counts=ltrdm_activities.hit_counts,
            click_counts=ltrdm_activities.click_counts,
            maay=ltrdm_activities.maay,
            grank=ltrdm_activities.grank,
            model=model,
            train_records=train_records,
            vali_records=vali_records
        )

    return activities



###########################
# CODE FOR ABLATION STUDY #
###########################

def prepare_ltr_rank(clicklogs: list[UserActivity], activities: list[UserActivity]) -> tuple[list[ClickThroughRecord], list[ClickThroughRecord], list[ClickThroughRecord]]:
    qid_mappings = {qid_key(ua) for ua in clicklogs} | {qid_key(ua) for ua in activities}
    
    unique_documents = {doc.infohash: doc for ua in clicklogs + activities for doc in ua.results}.values()
    corpus = {doc.infohash: doc.torrent_info.title.lower() for doc in unique_documents}
    corpus = Corpus(corpus)

    # create train.txt and vali.txt
    ltrdm_clicklogs = LTRDatasetMaker(clicklogs, comprehensive=False)
    ltrdm_clicklogs.corpus = corpus
    ltrdm_clicklogs.qid_mappings = qid_mappings
    records = ltrdm_clicklogs.compile_records()
    train_records, vali_records, _ = split_dataset_by_qids(records, train_ratio=0.8, val_ratio=0.2)
    del records
    
    # create test.txt
    ltrdm_activities = LTRDatasetMaker(activities, comprehensive=False)
    ltrdm_activities.corpus = corpus
    ltrdm_activities.qid_mappings = qid_mappings
    ltrdm_activities.hit_counts = compute_hit_counts(clicklogs)
    ltrdm_activities.click_counts = compute_click_counts(clicklogs)
    # ltrdm_activities.maay = MAAY(clicklogs)
    # ltrdm_activities.grank = precompute_grank_score_fn(clicklogs)
    test_records = ltrdm_activities.compile_records()
    
    return train_records, vali_records, test_records

def masked_ltr_rank(
         activities: list[UserActivity],
         train_records: list[ClickThroughRecord], 
         vali_records: list[ClickThroughRecord], 
         test_records: list[ClickThroughRecord], 
         masked_features: list[str]):
    
    for record in train_records + vali_records + test_records:
        record.qdr.mask(masked_features)

    config = Config.from_json("./allRank_config.json")
    dataset_path = f'.tmp/{uuid.uuid4().hex}/'

    write_records(dataset_path, {
        "train": train_records,
        "vali": vali_records,
        "test": test_records
    })
    normalize_features(dataset_path)
    config.data.path = os.path.join(dataset_path, "_normalized")

    try:
        model = create_trained_model(deepcopy(config), training=True)

        test_ds = load_libsvm_dataset_role("test", config.data.path, config.data.slate_length)
        test_dl = DataLoader(test_ds, batch_size=config.data.batch_size, num_workers=config.data.num_workers)
        
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
                    slate_scores = scores[i]
                    slate_indices = indices[i]
                    slate_mask = mask[i]
                    
                    valid_scores = slate_scores[~slate_mask]
                    valid_indices = slate_indices[~slate_mask]
                    
                    # Compute the rankings
                    _, sorted_idx = torch.sort(valid_scores, descending=True)
                    sorted_original_indices = valid_indices[sorted_idx]
                    sorted_indices = sorted_original_indices.cpu().tolist()
                    prev_results = activities[activity_idx].results
                    activities[activity_idx].results = [
                        prev_results[i] for i in sorted_indices
                    ]
                    
                    activity_idx += 1
    except Exception as e:
        raise e
    finally:
        shutil.rmtree(dataset_path, ignore_errors=True)

    return activities
