print("Importing modules...")
import pickle
import numpy as np
from argparse import ArgumentParser
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
import random
from baselines.panache import panache_rank
from baselines.dinx import dinx_rank, dinx_rank_by_seeders
from baselines.maay import maay_rank_numpy as maay_rank
from baselines.grank import grank_fast as grank
from baselines.random import random_rank
from baselines.tribler import tribler_rank
from baselines.ltr import ltr_rank
from common import UserActivity, mean_map, timing
print("Done importing modules")

np.random.seed(42)

K_RANGE = [5, 10, 30, None]
NUM_CORES = cpu_count()

def partition_list_uniform_sum(lst, num_partitions):
    """
    Partitions a list into `num_partitions` chunks such that the sum of values in each chunk is as equal as possible.

    :param lst: List of numbers to partition
    :param num_partitions: Number of partitions
    :return: List of partitions (each partition is a list)
    """
    total_sum = sum(lst)
    target_sum = total_sum / num_partitions  # Compute ideal sum per partition

    partitions = []
    current_partition = []
    current_sum = 0

    for num in lst:
        if current_sum + num > target_sum and current_partition:
            # If adding `num` exceeds target, finalize current partition
            partitions.append(current_partition)
            current_partition = []
            current_sum = 0
        
        # Add number to current partition
        current_partition.append(num)
        current_sum += num

    # Add the last partition
    if current_partition:
        partitions.append(current_partition)

    return partitions

# generate a range of indices with increasing steps
def gen_fast_range(end):
    total_range = []
    i = 0
    while i < end:
        total_range.append(i)
        i = int(i * 1.1) + 1
    total_range.append(end)
    return total_range

def chronological_eval(user_activities: list[UserActivity], rank_func, partition_id: int, total_partitions: int) -> dict[int, dict[int, float]]:
    TEST_DATA_SIZE = 100
    if rank_func.__name__ == "ltr_rank":
        NUM_WORKERS = min(NUM_CORES, 4)
        total_range = gen_fast_range(len(user_activities) - TEST_DATA_SIZE)
        total_range = partition_list_uniform_sum(total_range, total_partitions)[partition_id]
        print(f"Partition ID: {partition_id}, Total range: {total_range}")
    else:
        NUM_WORKERS = min(NUM_CORES, 8)
        total_range = range(0, len(user_activities) - TEST_DATA_SIZE)

    def process_index(i):
        try:
            # Use array indexing instead of slicing to avoid copying
            context = user_activities[:i]
            # Create indices for random selection instead of copying data
            test_indices = random.sample(range(i, len(user_activities)), TEST_DATA_SIZE)
            test_data = [user_activities[idx] for idx in test_indices]
            
            ranked_user_activities = rank_func(context, test_data)
            maps = {k: mean_map(ranked_user_activities, k) for k in K_RANGE}
            return i, maps
        except Exception as e:
            print(f"Error processing index {i}: {str(e)}")
            return None

    results = Parallel(n_jobs=NUM_WORKERS, batch_size=4)(
        delayed(process_index)(i) for i in tqdm(total_range, desc="Processing indices")
    )
    
    # Filter out None results and sort by context size
    results = [r for r in results if r is not None]
    results.sort(key=lambda x: x[0])
    
    return {i: maps for i, maps in results}

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-n', type=int, help='Number of user activities to use')
    parser.add_argument('--ltr', action='store_true', help='Only run LTR algorithm')
    parser.add_argument('--partition-id', type=int, help='Partition ID')
    parser.add_argument('--total-partitions', type=int, help='Total number of partitions')
    args = parser.parse_args()

    print(f"Number of CPU cores available: {NUM_CORES}")

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
        "panache": panache_rank,
        "dinx": dinx_rank,
        "dinx_s": dinx_rank_by_seeders,
        "maay": maay_rank,
        "grank": grank
    }
    if args.ltr:
        ranking_algos = {"dart": ltr_rank}

    all_results = {}

    for algo_name, ranking_algo in ranking_algos.items():
        print(f"============{algo_name}=============")
        
        with timing() as t:        
            maps = chronological_eval(user_activities, ranking_algo, args.partition_id, args.total_partitions)
            all_results[algo_name] = maps
    
    if args.ltr:
        with open(f'results/context_mrrs_ltr_{args.partition_id}_of_{args.total_partitions}.pkl', 'wb') as f:
            pickle.dump(all_results, f)
    else:
        with open(f'results/context_mrrs.pkl', 'wb') as f:
            pickle.dump(all_results, f)
