import pickle
from baselines.ltr import ltr_rank
import numpy as np
from common import mean_ndcg
from collections import defaultdict
from tqdm import tqdm
import json
from baselines.panache import panache_rank
from baselines.dinx import dinx_rank, dinx_rank_by_seeders
from baselines.maay import maay_rank_numpy as maay_rank
from baselines.grank import grank_fast as grank
from baselines.random import random_rank
from baselines.tribler import tribler_rank
from joblib import Parallel, delayed
np.random.seed(42)

K_RANGE = [5, 10, 30, None]

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

if __name__ == "__main__":

    print("Loading user activities...")
    with open('user_activities.pkl', 'rb') as f:
        user_activities = pickle.load(f)
    
    # Group activities by issuer
    print("Grouping activities by issuer...")
    issuer_groups = {}
    for activity in user_activities:
        if len(activity.results) < 30: 
            continue
        issuer = activity.issuer
        if issuer not in issuer_groups:
            issuer_groups[issuer] = []
        issuer_groups[issuer].append(activity)

    # Sort issuers by number of activities and keep those with at least 10 activities
    sorted_issuers = [(issuer, activities) for issuer, activities in issuer_groups.items() if len(activities) >= 10]
    sorted_issuers.sort(key=lambda x: len(x[1]), reverse=True)

    # Create new filtered list with just the top 32 users' activities
    grouped_activities = defaultdict(list)
    for issuer, activities in sorted_issuers:
        np.random.shuffle(activities)
        grouped_activities[issuer] = activities

    # Evaluate all activities for reference
    all_clicklogs = []
    for issuer, activities in tqdm(grouped_activities.items()):
        split_idx = int(len(activities) * 0.9)
        all_clicklogs.extend(activities[:split_idx])

    print(f"Total clicklogs: {len(all_clicklogs)}")

    algo_results = {}

    for algo_name, ranking_algo in ranking_algos.items():
        # Group activities by issuer for per-user evaluation
        results = []  # Store results for each user
        issuer_ndcgs = defaultdict(lambda: defaultdict(float))

        if algo_name == "ltr":
            print("Precomputing LTR for all clicklogs...")
            _, prec_data = ranking_algo(all_clicklogs, all_clicklogs[:1], precompute=True)
            print("Precomputing LTR done!")

        def process_user(issuer_activities):
            issuer, activities = issuer_activities
            split_idx = int(len(activities) * 0.9)
            
            user_result = {
                "num_clicklogs": len(activities),
                "split_idx": split_idx,
                "ndcg": {},
            }

            if algo_name == "ltr":
                user_reranked = ranking_algo(all_clicklogs, activities[split_idx:], prec_data=prec_data)
            else:
                user_reranked = ranking_algo(all_clicklogs, activities[split_idx:])
            
            for k in K_RANGE:
                user_ndcgs = mean_ndcg(user_reranked, k=k)
                user_result["ndcg"][k] = float(np.mean(user_ndcgs))
                print(f'User {issuer} ({len(activities)}) nDCG@{k}: {user_result["ndcg"][k]}')
            
            return user_result

        results = Parallel(n_jobs=4)(
            delayed(process_user)(item) for item in tqdm(grouped_activities.items())
        )
        
        algo_results[algo_name] = results

    # Save results to JSON file
    with open('results/p2p.json', 'w') as f:
        json.dump(algo_results, f)
