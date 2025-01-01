import pickle
from sklearn.metrics import ndcg_score
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from baselines.panache import panache_rank, panache_rank_fast
from baselines.dinx import dinx_rank
from baselines.maay import maay_rank, maay_rank_numpy
from baselines.grank import grank
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

    user_activities.sort(key=lambda ua: ua.timestamp)

    print("============Tribler=============")
    print(f"Average nDCG: {mean_ndcg(user_activities)}")

    print("============Random=============")
    for ua in user_activities:
        np.random.shuffle(ua.results)
    print(f"Average nDCG: {mean_ndcg(user_activities)}")

    print("============Panaché=============")
    start = time.time()
    ranked_user_activities = panache_rank(user_activities)
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")
    print(f"Average nDCG: {mean_ndcg(ranked_user_activities)}")

    print("============Panaché Fast=============")
    start = time.time()
    ranked_user_activities = panache_rank_fast(user_activities)
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")
    print(f"Average nDCG: {mean_ndcg(ranked_user_activities)}")


    
    # ltr_rank(user_activities)

    # average_ndcg = np.mean([calculate_ndcg(ua) for ua in user_activities])
    # print(f"Average nDCG: {average_ndcg}")