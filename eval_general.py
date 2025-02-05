print("Importing modules...")
import pickle
import os
import numpy as np
from argparse import ArgumentParser
from baselines.panache import panache_rank
from baselines.dinx import dinx_rank, dinx_rank_by_seeders
from baselines.maay import maay_rank_numpy as maay_rank
from baselines.grank import grank_fast as grank
from baselines.random import random_rank
from baselines.tribler import tribler_rank
from baselines.ltr import ltr_rank
from common import mean_ndcg, mean_map, timing
print("Done importing modules")

np.random.seed(123)

K_RANGE = [5, 10, 30, None]

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-n', type=int, help='Number of user activities to use')
    args = parser.parse_args()

    print("Loading user activities...")
    with open('user_activities.pkl', 'rb') as f:
        user_activities = pickle.load(f)
    np.random.shuffle(user_activities)
    if args.n:
        user_activities = user_activities[:args.n]
    print(f"Loaded {len(user_activities)} user activities.")

    ranking_algos = {
        "tribler": tribler_rank,
        "random": random_rank,
        "ltr": ltr_rank,
        "panache": panache_rank,
        "dinx": dinx_rank,
        "dinx_s": dinx_rank_by_seeders,
        "maay": maay_rank,
        "grank": grank
    }

    split_idx = int(0.9 * len(user_activities))
    results = {}
    for k in K_RANGE:
        results[f"ndcg@{k}"] = {}
        results[f"map@{k}"] = {}
    

    activities_dump_dir = 'results/reranked_activities'
    os.makedirs(activities_dump_dir, exist_ok=True)

    for algo_name, ranking_algo in ranking_algos.items():
        print(f"============{algo_name}=============")
        
        with timing() as t:
            reranked_activities = ranking_algo(
                user_activities[:split_idx],  
                user_activities[split_idx:]
            )

        for k in K_RANGE:
            ndcgs = mean_ndcg(reranked_activities, k=k)
            results[f"ndcg@{k}"][algo_name] = np.mean(ndcgs)
            print(f"NDCG@{k}: {np.mean(ndcgs)}")

        for k in K_RANGE:
            maps = mean_map(reranked_activities, k=k)
            results[f"map@{k}"][algo_name] = np.mean(maps)
            print(f"MAP@{k}: {np.mean(maps)}")
        
        with open(os.path.join(activities_dump_dir, f'{algo_name}.pkl'), 'wb') as f:
            pickle.dump(reranked_activities, f)

    with open('results/general_ndcg.tsv', 'w') as f:
        f.write('metric\t' + '\t'.join(ranking_algos.keys()) + '\n')
        for k in K_RANGE:
            k_str = 'max' if k is None else str(k)
            f.write(f'NDCG@{k_str}')
            for algo_name in ranking_algos:
                f.write(f'\t{results[f"ndcg@{k}"][algo_name]:.4f}')
            f.write('\n')
        for k in K_RANGE:
            k_str = 'max' if k is None else str(k)
            f.write(f'MAP@{k_str}')
            for algo_name in ranking_algos:
                f.write(f'\t{results[f"map@{k}"][algo_name]:.4f}')
            f.write('\n')
    
    print('Success!')
