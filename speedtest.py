import pickle
from sklearn.metrics import ndcg_score
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from baselines.panache import panache_rank
from baselines.dinx import dinx_rank, dinx_rank_by_seeders
from baselines.maay import maay_rank_numpy as maay_rank
from baselines.grank import grank_fast as grank
from baselines.random import random_rank
from baselines.tribler import tribler_rank
from baselines.ltr import ltr_rank
import time

np.random.seed(42)

def calc_ndcg(ua):
    true_relevance = [1 if res.infohash == ua.chosen_result.infohash else 0 for res in ua.results]
    predicted_relevance = [1/np.log2(i+2) for i in range(len(ua.results))]
    return ndcg_score([true_relevance], [predicted_relevance])

def simulate_gradually(user_activities, rank_func):
    # Move the processing logic to a separate function
    def process_index(i, user_activities, rank_func):
        ranked_user_activities = rank_func(user_activities[:i], user_activities[i:i+1])
        return mean_ndcg(ranked_user_activities)

    # Run parallel computation with tqdm progress bar
    ndcgs = Parallel(n_jobs=-1)(
        delayed(process_index)(i, user_activities, rank_func)
        for i in tqdm(range(len(user_activities)), desc="Simulating gradually")
    )
    
    return np.mean(ndcgs)

def mean_ndcg(user_activities):
    return np.mean([calc_ndcg(ua) for ua in user_activities])

if __name__ == "__main__":
    with open('user_activities.pkl', 'rb') as f:
        user_activities = pickle.load(f)
    user_activities = user_activities[:100]

    ranking_algos = {
        "Tribler": tribler_rank, # must be first
        "Random": random_rank,
        "LTR": ltr_rank,
        "Panaché": panache_rank,
        "DINX": dinx_rank,
        "DINX (seeders)": dinx_rank_by_seeders,
        "MAAY": maay_rank,
        "G-Rank": grank
    }

    all_ndcgs = {}
    split_idx = int(0.8 * len(user_activities))
    np.random.shuffle(user_activities)

    for algo_name, ranking_algo in ranking_algos.items():
        print(f"============{algo_name}=============")

        start_time = time.time()
        reranked_activities = ranking_algo(
            user_activities[:split_idx],
            user_activities[split_idx:]
        )
        elapsed_time = time.time() - start_time
        print(f"Ranking took {elapsed_time:.2f} seconds")


    
    # ltr_rank(user_activities)

    # average_ndcg = np.mean([calculate_ndcg(ua) for ua in user_activities])
    # print(f"Average nDCG: {average_ndcg}")