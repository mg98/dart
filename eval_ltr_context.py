print("Importing modules...")
import pickle
import numpy as np
from argparse import ArgumentParser
from allrank.config import Config
from joblib import cpu_count
from tqdm import tqdm
from baselines.ltr import ltr_rank
from common import UserActivity, mean_map, timing
print("Done importing modules")

np.random.seed(42)

K_RANGE = [5, 10, 30, None]
NUM_CORES = cpu_count()
TEST_SIZE = 1000

if __name__ == "__main__":
    config = Config.from_json("./allRank_config.json")
    config.data.path = ''

    parser = ArgumentParser()
    parser.add_argument('-n', type=int, help='Number of user activities to use')
    parser.add_argument('--size', type=int, help='Context size')
    args = parser.parse_args()

    print("Loading user activities...")
    with open('user_activities.pkl', 'rb') as f:
        user_activities = pickle.load(f)
    np.random.shuffle(user_activities)
    if args.n:
        user_activities = user_activities[:args.n]
    if args.size > len(user_activities) - TEST_SIZE:
        raise ValueError("Context size too large")
    

    print(f"Loaded {len(user_activities)} user activities.")
    print(f"Running experiment with context size: {args.size}")

    print(f"============DART=============")
    with timing() as t:
        ranked_user_activities = ltr_rank(
            user_activities[:args.size],
            user_activities[-TEST_SIZE:],
            config
        )
    
    mrr = mean_map(ranked_user_activities)
    print(f"MRR: {mrr}")
    with open('./results/ltr_context_mrr.tsv', 'a+') as f:
        f.write(f"{args.size}\t{mrr}\n")

    print('Success!')
