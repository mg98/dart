import sys
import os

# Add the path to the allrank submodule
sys.path.append('../allRank')

from allrank.models.losses import listMLE
from allrank.models.model import make_model
from allrank.data.dataset import ListwiseDataset
from allrank.config import Config
from allrank.training.train_utils import fit
from allrank.models.metrics import ndcg

import torch
from torch.utils.data import DataLoader
import numpy as np
from common import UserActivity

def gen_ltr_dataset(user_activities: list[UserActivity]):
    
    pass

def ltr_rank(user_activities: list[UserActivity]):
    """
    Implementing Learning to Rank using the allRank library.
    """
    # Prepare data
    X = []
    y = []
    for ua in user_activities:
        features = [[t.seeders, t.leechers] for t in ua.results]
        X.append(features)
        relevance = [1 if i == ua.chosen_index else 0 for i in range(len(ua.results))]
        y.append(relevance)

    X = np.array(X)
    y = np.array(y)

    # Create dataset and dataloader
    dataset = ListwiseDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Configure model
    config = Config(
        model=dict(
            fc_model=dict(sizes=[64, 32, 1], input_size=2, activation="ReLU", dropout=0.1),
            transformer=None
        ),
        loss=dict(name="listMLE"),
        optimizer=dict(name="Adam", lr=0.001),
        lr_scheduler=dict(name="StepLR", step_size=100, gamma=0.1),
        epochs=10
    )

    # Create and train model
    model = make_model(config)
    fit(model, dataloader, config, val_dataloader=None)

    # Re-rank results
    model.eval()
    with torch.no_grad():
        for ua in user_activities:
            features = torch.FloatTensor([[t.seeders, t.leechers] for t in ua.results])
            scores = model(features.unsqueeze(0)).squeeze()
            sorted_indices = torch.argsort(scores, descending=True)
            ua.results = [ua.results[i] for i in sorted_indices]

    return user_activities

