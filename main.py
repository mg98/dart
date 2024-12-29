import pickle
import sys
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score
from baselines.panache import panache_rank_fast as panache_rank
from baselines.dinx import dinx_rank, dinx_rank_by_seeders
from baselines.maay import maay_rank_numpy as maay_rank
from baselines.grank import grank_fast as grank
from baselines.random import random_rank
from baselines.tribler import tribler_rank
from baselines.ltr import ltr_rank

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

def simulate_gradually(user_activities, rank_func, batch_size=64, k=10):
    user_activities.sort(key=lambda ua: ua.timestamp)

    # Move the processing logic to a separate function
    def process_batch(i, user_activities, rank_func):
        batch_start = i * batch_size
        batch_end = min(batch_start + batch_size, len(user_activities))
        ranked_user_activities = rank_func(user_activities[:batch_start], user_activities[batch_start:batch_end])
        return mean_ndcg(ranked_user_activities, k)

    # Run parallel computation with tqdm progress bar
    ndcgs = Parallel(n_jobs=-1)(
        delayed(process_batch)(i, user_activities, rank_func)
        for i in tqdm(range(len(user_activities) // batch_size), desc="Processing batches")
    )
    
    return ndcgs

def parallel_batched_processing(user_activities, ranking_fn, batch_size=64, k=10):
    """Compute leave-n-out nDCG scores for a ranking function"""
    def process_batch(i):
        reranked_activities = ranking_fn(
            user_activities[:i*batch_size] + user_activities[(i+1)*batch_size:], 
            user_activities[i*batch_size:(i+1)*batch_size]
            )
        return mean_ndcg(reranked_activities, k=k)

    np.random.shuffle(user_activities)
    ndcgs = Parallel(n_jobs=-1)(
        delayed(process_batch)(i)
        for i in range(len(user_activities) // batch_size)
    )
    return ndcgs

def mean_ndcg(user_activities, k=None):
    return np.round(np.mean([calc_ndcg(ua, k) for ua in user_activities]), 3)

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
    with open('user_activities.pkl', 'rb') as f:
        user_activities = pickle.load(f)

    ranking_algos = {
        "Tribler": tribler_rank,
        "Random": random_rank,
        "Panaché": panache_rank,
        "DINX": dinx_rank,
        "DINX (seeders)": dinx_rank_by_seeders,
        "MAAY": maay_rank,
        "G-Rank": grank
    }

    all_ndcgs = {}
    
    for algo_name, ranking_algo in ranking_algos.items():
        print(f"============{algo_name}=============")

        for k in K_RANGE:
            ndcgs = parallel_batched_processing(user_activities, ranking_algo, k=k)
            print(f"nDCG@{k}: {np.mean(ndcgs)}")

        ndcgs = simulate_gradually(user_activities, ranking_algo, batch_size=1)
        all_ndcgs[algo_name] = ndcgs
    
    plot_ndcg(all_ndcgs, filename='combined_results.png')

    # print("============DINX=============")
    # for k in K_RANGE:
    #     ndcgs = parallel_batched_processing(user_activities, dinx_rank, k=k)
    #     print(f"nDCG@{k}: {np.mean(ndcgs)}")

    # # ranked_user_activities = dinx_rank(user_activities)
    # # for k in K_RANGE:
    # #     print(f"nDCG@{k}: {mean_ndcg(ranked_user_activities, k=k)}")
    # # ndcgs = simulate_gradually(user_activities, dinx_rank)
    
    # # Plot the gradual nDCG scores
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(len(ndcgs)), ndcgs)
    # plt.xlabel('Number of Training Examples')
    # plt.ylabel('nDCG Score')
    # plt.title('DINX Learning Curve')
    # plt.grid(True)
    # plt.savefig('dinx_learning_curve.png')
    # plt.close()
    # #print(f"nDCG: {simulate_gradually(user_activities, dinx_rank)}")

    # print("============DINX by seeders=============")
    # for k in K_RANGE:
    #     ndcgs = parallel_batched_processing(user_activities, dinx_rank_by_seeders, k=k)
    #     print(f"nDCG@{k}: {np.mean(ndcgs)}")
    # #print(f"nDCG: {simulate_gradually(user_activities, dinx_rank_by_seeders)}")

    # print("============MAAY=============") 
    # ranked_user_activities = maay_rank(user_activities)
    # for k in K_RANGE:
    #     ndcgs = parallel_batched_processing(user_activities, maay_rank, k=k)
    #     print(f"nDCG@{k}: {np.mean(ndcgs)}")
    # #print(f"nDCG: {simulate_gradually(user_activities, maay_rank)}")

    # print("============G-Rank=============")
    # ranked_user_activities = grank(user_activities)
    # for k in K_RANGE:
    #     ndcgs = parallel_batched_processing(user_activities, grank, k=k)
    #     print(f"nDCG@{k}: {np.mean(ndcgs)}")
    #print(f"nDCG: {simulate_gradually(user_activities, grank)}")
    
    ltr_rank(user_activities)

    # average_ndcg = np.mean([calculate_ndcg(ua) for ua in user_activities])
    # print(f"Average nDCG: {average_ndcg}")