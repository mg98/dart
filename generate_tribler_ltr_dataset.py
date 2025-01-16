import re
from rank_bm25 import BM25Okapi
import pickle
import numpy as np
import pandas as pd
import shutil
from collections import defaultdict
from tqdm import tqdm
from common import *
from baselines.panache import compute_hit_counts
from ltr_helper import LTRDatasetMaker

np.random.seed(42)

EXPORT_PATH = "tribler_data"
shutil.rmtree(EXPORT_PATH, ignore_errors=True)
os.makedirs(EXPORT_PATH, exist_ok=True)

with open('user_activities.pkl', 'rb') as f:
    user_activities = pickle.load(f)

################
# MAIN DATASET #
################

print("Generating main dataset...")

ltrdm = LTRDatasetMaker(user_activities)
ltrdm.generate(EXPORT_PATH)
ltrdm.write_queries(EXPORT_PATH)

#########################
# USER-SPECIFIC DATASET #
#########################

print("Generating user-specific datasets...")

# Get top users by number of queries
user_query_counts = ltrdm.df.groupby('user')['query'].nunique()
top_users = user_query_counts.nlargest(32).index
original_df = ltrdm.df.copy()

# Create per-user datasets
for user in tqdm(top_users):
    # Get qids for this user's queries
    individual_user_activity_df = original_df[original_df['user'] == user]
    individual_user_activity_df = individual_user_activity_df.sort_values('timestamp')
    ltrdm.df = individual_user_activity_df
    
    # Create user directory
    user_dir = os.path.join(EXPORT_PATH, "by_user", str(user))
    os.makedirs(user_dir, exist_ok=True)

    # Filter records for this user's queries
    user_records = ltrdm.generate(user_dir, normalize=False)
    
# Create temporary combined files for normalization
temp_dir = os.path.join(EXPORT_PATH, "by_user", "temp")
os.makedirs(temp_dir, exist_ok=True)

try:
    # Combine all user files
    for file_type in ['train.txt', 'vali.txt', 'test.txt']:
        with open(os.path.join(temp_dir, file_type), 'w') as outfile:
            for user in top_users:
                user_dir = os.path.join(EXPORT_PATH, "by_user", str(user))
                with open(os.path.join(user_dir, file_type)) as infile:
                    outfile.write(infile.read())

    # Run normalization on combined files
    normalize_features(temp_dir)

    # Split normalized files back into user-specific files
    for file_type in ['train.txt', 'vali.txt', 'test.txt']:
        # Read the full normalized file
        with open(os.path.join(temp_dir, '_normalized', file_type)) as f:
            normalized_lines = f.readlines()
        
        # Track current position in normalized lines
        pos = 0
        
        # Split back into user files
        for user in top_users:
            user_dir = os.path.join(EXPORT_PATH, "by_user", str(user))
            norm_dir = os.path.join(user_dir, '_normalized')
            os.makedirs(norm_dir, exist_ok=True)
            
            # Count lines in original user file
            with open(os.path.join(user_dir, file_type)) as f:
                line_count = sum(1 for _ in f)
                
            # Write that many normalized lines to user's normalized file
            with open(os.path.join(norm_dir, file_type), 'w') as f:
                f.writelines(normalized_lines[pos:pos+line_count])
            pos += line_count

except Exception as e:
    raise e

finally:
    shutil.rmtree(temp_dir)

# Assert no empty dataset files exist
by_user_dir = os.path.join(EXPORT_PATH, "by_user")
for root, dirs, files in os.walk(by_user_dir):
    for file in files:
        if file in ['train.txt', 'vali.txt', 'test.txt']:
            file_path = os.path.join(root, file)
            assert os.path.getsize(file_path) > 0, f"Found empty dataset file: {file_path}"
