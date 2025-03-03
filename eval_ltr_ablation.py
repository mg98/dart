print("Importing modules...")
import pickle
import os
import numpy as np
from argparse import ArgumentParser
from baselines.ltr import prepare_ltr_rank, masked_ltr_rank
from common import mean_mrr
from copy import deepcopy
print("Done importing modules")

np.random.seed(42)

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

    split_idx = int(0.9 * len(user_activities))

    clicklogs = user_activities[:split_idx]
    activities = user_activities[split_idx:]

    for ua in clicklogs:
        np.random.shuffle(ua.results)
    for ua in activities:
        np.random.shuffle(ua.results)
    np.random.shuffle(clicklogs)
    np.random.shuffle(activities)

    with open('tribler_data/test_activities.pkl', 'rb') as f:
        activities = pickle.load(f)

    train_records, vali_records, test_records = prepare_ltr_rank(
        clicklogs,  
        activities
    )

    with open('results/ablation_study.tsv', 'w') as f:
        f.write('masked\tMRR\n')

        for mask in [
            [],
            ["title"],
            ["seeders"],
            ["leechers"],
            ["click_count"],
            ["query_hit_count"],
            # ["sp", "rel", "pop", "matching_score"],
            # ["grank_score"],
            ["pos"],
            ["tag_count"],
            ["age"],
        ]:
            print(f"Masking {mask}")
            reranked_activities = masked_ltr_rank(
                deepcopy(activities),
                deepcopy(train_records),
                deepcopy(vali_records),
                deepcopy(test_records),
                mask
            )
            f.write(f'{",".join(mask)}')
            mrr = mean_mrr(reranked_activities)
            print(f"MRR: {mrr}")
            f.write(f'\t{mrr:.4f}')
            f.write('\n')
        
        print('Success!')
