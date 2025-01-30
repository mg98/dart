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

with open('user_activities.pkl', 'rb') as f:
    user_activities = pickle.load(f)

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

# Group activities by issuer
issuer_groups = {}
for activity in user_activities:
    issuer = activity.issuer
    if issuer not in issuer_groups:
        issuer_groups[issuer] = []
    issuer_groups[issuer].append(activity)

# Sort issuers by number of activities and keep top 32
sorted_issuers = sorted(issuer_groups.items(), key=lambda x: len(x[1]), reverse=True)[:32]

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

algo_results = {}

for algo_name, ranking_algo in ranking_algos.items():
    # Group activities by issuer for per-user evaluation
    results = []  # Store results for each user
    issuer_ndcgs = defaultdict(lambda: defaultdict(float))
    for issuer, activities in tqdm(grouped_activities.items()):
        print(f"\nEvaluating user {issuer} ({len(activities)} activities)")
        
        # Split user's activities 90:10
        split_idx = int(len(activities) * 0.9)
        user_reranked = ranking_algo(activities[:split_idx], activities[split_idx:])
        user_reranked_gossip = ranking_algo(all_clicklogs, activities[split_idx:])
        
        # Calculate nDCG@k for this user
        user_result = {
            "num_clicklogs": len(activities),
            "split_idx": split_idx,
            "local_ndcg": {},
            "gossip_ndcg": {},
        }
        
        for k in K_RANGE:
            user_ndcgs = mean_ndcg(user_reranked, k=k)
            user_result["local_ndcg"][k] = float(np.mean(user_ndcgs))

            user_ndcgs = mean_ndcg(user_reranked_gossip, k=k)
            user_result["gossip_ndcg"][k] = float(np.mean(user_ndcgs))

            print(f'User {issuer} nDCG@{k}: {user_result["local_ndcg"][k]} (local), {user_result["gossip_ndcg"][k]} (gossip)')
        
        results.append(user_result)
    
    algo_results[algo_name] = results

# Save results to JSON file
with open('results/p2p.json', 'w') as f:
    json.dump(results, f)

