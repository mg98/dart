print("Importing modules...")
import pickle
import os
import numpy as np
from argparse import ArgumentParser
from allrank.config import Config
from baselines.panache import panache_rank
from baselines.dinx import dinx_rank, dinx_rank_by_seeders
from baselines.maay import maay_rank_numpy as maay_rank
from baselines.grank import grank_fast as grank
from baselines.random import random_rank
from baselines.tribler import tribler_rank
from baselines.ltr import ltr_rank
from utils.common import mean_mrr, timing
print("Done importing modules")

np.random.seed(42)

if __name__ == "__main__":
    config = Config.from_json("./allRank_config.json")

    parser = ArgumentParser()
    parser.add_argument('-n', type=int, help='Number of user activities to use')
    args = parser.parse_args()

    print("Loading user activities...")
    with open('user_activities.pkl', 'rb') as f:
        user_activities = pickle.load(f)
    np.random.shuffle(user_activities)
    if args.n:
        user_activities = user_activities[:args.n]
        config.data.path = ''
    print(f"Loaded {len(user_activities)} user activities.")

    ranking_algos = {
        "tribler": tribler_rank,
        "random": random_rank,
        "panache": panache_rank,
        "dinx": dinx_rank,
        "dinx_s": dinx_rank_by_seeders,
        "maay": maay_rank,
        "grank": grank,
        "dart": ltr_rank
    }

    split_idx = int(0.9 * len(user_activities))

    activities_dump_dir = 'results/reranked_activities'
    os.makedirs(activities_dump_dir, exist_ok=True)

    for algo_name, ranking_algo in ranking_algos.items():
        print(f"============{algo_name}=============")
        
        with timing() as t:
            if algo_name == "ltr":
                reranked_activities = ranking_algo(
                    user_activities[:split_idx],  
                    user_activities[split_idx:],
                    config
                )
            else:
                reranked_activities = ranking_algo(
                    user_activities[:split_idx],  
                    user_activities[split_idx:]
                )

        mrr = mean_mrr(reranked_activities)
        print(f"MRR: {mrr}")
        
        with open(os.path.join(activities_dump_dir, f'{algo_name}.pkl'), 'wb') as f:
            pickle.dump(reranked_activities, f)
    
    print('Success!')
