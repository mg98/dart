print("Importing modules...")
import pickle
import numpy as np
import time
from joblib import Parallel, delayed
from tqdm import tqdm
from baselines.panache import panache_rank
from baselines.dinx import dinx_rank, dinx_rank_by_seeders
from baselines.maay import maay_rank_numpy as maay_rank
from baselines.grank import grank_fast as grank
from baselines.random import random_rank
from baselines.tribler import tribler_rank
from baselines.ltr import ltr_rank
from common import UserActivity, mean_ndcg, timing
print("Done importing modules")

np.random.seed(43)

K_RANGE = [5, 10, 30, None]

def chronological_eval(user_activities: list[UserActivity], rank_func) -> dict[int, list[float]]:
    user_activities.sort(key=lambda ua: ua.timestamp)
    test_data_size = 100

    # first index at which 5 distinct queries are seen (required to take 0.8/0.2 splits)
    offset = next(
        idx for idx, _ in enumerate(user_activities)
        if len({(a.query, a.issuer) for a in user_activities[:idx]}) == 5
    )

    NUM_WORKERS = 4 if rank_func.__name__ == "ltr_rank" else 8
    STEP_SIZE = 1
    total_range = range(offset, len(user_activities) - test_data_size, STEP_SIZE)

    # Split range into NUM_JOBS parts based on modulo
    NUM_CHUNKS = NUM_WORKERS * max(1, len(user_activities) // 500)
    range_chunks = [
        [i for i in total_range if i % NUM_CHUNKS == chunk_id]
        for chunk_id in range(NUM_CHUNKS)
    ]

    def process_batch(i_range):
        batch_ndcgs = []
        position = i_range[0] % NUM_WORKERS
        for i in tqdm(i_range, 
                     desc=f"Worker #{position}", 
                     leave=True, 
                     position=position):
            try:
                test_data = list(np.random.choice(user_activities[i:], size=test_data_size, replace=False))
                ranked_user_activities = rank_func(user_activities[:i], test_data)
                ndcgs = {k: mean_ndcg(ranked_user_activities, k) for k in K_RANGE}
                batch_ndcgs.append(ndcgs)
            except Exception as e:
                print(f"Error processing index {i}: {str(e)}")
                continue
        return batch_ndcgs

    ndcgs = []
    all_indices = []  # Track original indices
    for batch_ndcgs, r in Parallel(n_jobs=NUM_WORKERS)(
            delayed(lambda x: (process_batch(x), x))(r) for r in tqdm(
                range_chunks,
                desc="Processing batches",
                )
        ):
        ndcgs.extend(batch_ndcgs)
        all_indices.extend(r)
    
    tqdm.write('')
    
    # Sort results by original indices
    ndcgs = [x for _, x in sorted(zip(all_indices, ndcgs))]

    # Reorganize ndcgs from list of dicts to dict of lists
    combined_ndcgs = {}
    for k in K_RANGE:
        combined_ndcgs[k] = [batch_ndcgs[k] for batch_ndcgs in ndcgs]
    ndcgs = combined_ndcgs

    return ndcgs

if __name__ == "__main__":
    print("Loading user activities...")
    with open('user_activities.pkl', 'rb') as f:
        user_activities = pickle.load(f)
    np.random.shuffle(user_activities)
    user_activities = user_activities
    print(f"Loaded {len(user_activities)} user activities.")

    ranking_algos = {
        "Tribler": tribler_rank,
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

    for algo_name, ranking_algo in ranking_algos.items():
        print(f"============{algo_name}=============")
        
        with timing() as t:
        
            # reranked_activities = ranking_algo(
            #     user_activities[:split_idx],
            #     user_activities[split_idx:]
            # )
            # for k in K_RANGE:
            #     ndcgs = mean_ndcg(reranked_activities, k=k)
            #     print(f"nDCG@{k}: {np.mean(ndcgs)}")
        
            ndcgs = chronological_eval(user_activities, ranking_algo)
            all_ndcgs[algo_name] = ndcgs
        
    with open('chronological_ndcgs.pkl', 'wb') as f:
        pickle.dump(all_ndcgs, f)

    print('Success!')
