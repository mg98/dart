print("Importing modules...")
import pickle
import os
import numpy as np
from argparse import ArgumentParser
from baselines.ltr import prepare_ltr_rank, masked_ltr_rank
from common import mean_ndcg
from copy import deepcopy
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

    split_idx = int(0.9 * len(user_activities))

    clicklogs = user_activities[:split_idx]
    activities = user_activities[split_idx:]

    for ua in clicklogs:
        np.random.shuffle(ua.results)
    for ua in activities:
        np.random.shuffle(ua.results)
    np.random.shuffle(clicklogs)
    np.random.shuffle(activities)

    train_records, vali_records, test_records = prepare_ltr_rank(
        clicklogs,  
        activities
    )

    with open('results/ablation_study.tsv', 'w') as f:
        f.write('masked\tNDCG@' + '\tNDCG@'.join(map(str, K_RANGE)) + '\n')

        for mask in [
            ["title"],
            ["seeders"],
            ["leechers"],
            ["click_count"],
            ["query_hit_count"],
            ["sp", "rel", "pop", "matching_score"],
            ["grank_score"],
            ["pos"],
            ["tag_count"],
            ["age"],
        ]:
            print(f"Masking {mask}")
            reranked_activities = masked_ltr_rank(
                deepcopy(activities),
                train_records,
                vali_records,
                test_records,
                mask
            )
            f.write(f'{",".join(mask)}')
            for k in K_RANGE:
                ndcgs = mean_ndcg(reranked_activities, k=k)
                print(f"NDCG@{k}: {np.mean(ndcgs)}")
                f.write(f'\t{np.mean(ndcgs):.4f}')
            f.write('\n')
        
        print('Success!')
