print("Importing modules...")
import pickle
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score
from baselines.panache import panache_rank
from baselines.dinx import dinx_rank, dinx_rank_by_seeders
from baselines.maay import maay_rank_numpy as maay_rank
from baselines.grank import grank_fast as grank
from baselines.random import random_rank
from baselines.tribler import tribler_rank
from baselines.ltr import ltr_rank
import contextlib
import os
print("Done importing modules")

np.random.seed(42)

K_RANGE = [5, 10, 30, 60]

def calc_mrr(ua):
    """Calculate Mean Reciprocal Rank for a single user activity"""
    for i, res in enumerate(ua.results):
        if res.infohash == ua.chosen_result.infohash:
            return 1.0 / (i + 1)
    return 0.0

def mean_mrr(user_activities):
    return np.mean([calc_mrr(ua) for ua in user_activities])

def calc_ndcg(ua, k=None):
    """Calculate nDCG@k for a single user activity
    Args:
        ua: user activity
        k: number of top results to consider. If None, considers all results
    """
    true_relevance = [1 if res.infohash == ua.chosen_result.infohash else 0 for res in ua.results]
    predicted_relevance = [1/np.log2(i+2) for i in range(len(ua.results))]
    
    if k is not None:
        # Truncate both lists to k elements
        true_relevance = true_relevance[:k]
        predicted_relevance = predicted_relevance[:k]
    
    return ndcg_score([true_relevance], [predicted_relevance])

def mean_ndcg(user_activities, k=None):
    return np.round(np.mean([calc_ndcg(ua, k) for ua in user_activities]), 3)

def chronological_eval(user_activities, rank_func, batch_size=64, k=10):
    user_activities.sort(key=lambda ua: ua.timestamp)

    # first index at which 5 distinct queries are seen (required to take 0.8/0.2 splits)
    offset = next(
        idx for idx, _ in enumerate(user_activities)
        if len({a.query for a in user_activities[:idx+1]}) == 5
    )

    # Move the processing logic to a separate function
    def process_batch(i, user_activities, rank_func):
        batch_start = 18#(i + 1) * batch_size
        batch_end = 19#min(batch_start + batch_size, len(user_activities))
        print("Batch start: ", batch_start, "Batch end: ", batch_end)
        ranked_user_activities = rank_func(user_activities[:batch_start], user_activities[batch_start:batch_end])
        return mean_ndcg(ranked_user_activities, k)

    # Run parallel computation with tqdm progress bar
    ndcgs = [
        process_batch(i, user_activities, rank_func)
        for i in tqdm(range(offset, len(user_activities) // batch_size - 1), desc="Processing batches")
    ]
    
    return ndcgs

def simple_eval(user_activities, ranking_fn, k=10):
    """
    Sample 80% of the data for training and 20% for testing.
    Then, rerank the training set and evaluate the test set.
    """
    np.random.shuffle(user_activities)
    split_idx = int(0.8 * len(user_activities))
    reranked_activities = ranking_fn(
        user_activities[:split_idx],
        user_activities[split_idx:]
    )
    return mean_ndcg(reranked_activities, k=k)

def plot_ndcg(ndcgs_dict, window_size=1000, filename='combined_plot.png'):
    plt.figure(figsize=(12, 8))
    
    for algo_name, ndcgs in ndcgs_dict.items():
        smoothed_ndcgs = np.convolve(ndcgs, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(len(smoothed_ndcgs)), smoothed_ndcgs, label=algo_name)
    
    plt.xlabel('Number of Training Examples')
    plt.ylabel('nDCG')
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    print("Loading user activities...")
    with open('user_activities.pkl', 'rb') as f:
        user_activities = pickle.load(f)
    user_activities = user_activities[:500]
    print(f"Loaded {len(user_activities)} user activities.")

    ranking_algos = {
        # "Tribler": tribler_rank, # must be first
        # "Random": random_rank,
        "LTR": ltr_rank,
        "Panaché": panache_rank,
        "DINX": dinx_rank,
        "DINX (seeders)": dinx_rank_by_seeders,
        "MAAY": maay_rank,
        "G-Rank": grank
    }

    all_ndcgs = {}
    split_idx = int(0.8 * len(user_activities))
    
    for algo_name, ranking_algo in ranking_algos.items():
        print(f"============{algo_name}=============")

        # for k in K_RANGE:
        #     reranked_activities = ranking_algo(
        #         user_activities[:split_idx],
        #         user_activities[split_idx:]
        #     )
        #     ndcgs = mean_ndcg(reranked_activities, k=k)
        #     print(f"nDCG@{k}: {np.mean(ndcgs)}")
        
        ndcgs = chronological_eval(user_activities, ranking_algo, batch_size=1)
        all_ndcgs[algo_name] = ndcgs

        # shuffle here, so tribler_rank gets original order
        np.random.shuffle(user_activities)
    
    plot_ndcg(all_ndcgs, filename='combined_results.png')
