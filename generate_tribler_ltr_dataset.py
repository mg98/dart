import json
import re
from rank_bm25 import BM25Okapi
import time
import numpy as np
import pandas as pd
import shutil
from common import *

np.random.seed(42)

EXPORT_PATH = "tribler_data"

def tokenize(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower().split()

def compile_records(df):
    records = []

    for _, row in df.iterrows():
        qid = list(unique_queries).index(row['query'])
        query_terms = tokenize(row['query'])
        doc_indices = [doc_ids.index(result.infohash) for result in row['results']]
        bm25_scores = bm25.get_batch_scores(query_terms, doc_indices)

        for i, result in enumerate(row['results']):

            record = ClickThroughRecord()
            record.qid = qid
            if isinstance(row['chosen_result'], str):
                record.rel = 1.0 if result.infohash == row['chosen_result'] else 0.0
            else:
                record.rel = row['chosen_result'].get(result.infohash, 0) / max(row['chosen_result'].values())

            v = QueryDocumentRelationVector()
            v.seeders = result.seeders
            v.leechers = result.leechers
            v.age = ua.timestamp - result.torrent_info.timestamp
            v.bm25 = bm25_scores[i]

            # aggregate tf idf features over all query terms
            tfidf_results = [tfidf.get_tf_idf(result.infohash, term) for term in query_terms]

            v.tf_min = min(r["tf"] for r in tfidf_results)
            v.tf_max = max(r["tf"] for r in tfidf_results)
            v.tf_sum = sum(r["tf"] for r in tfidf_results)
            v.tf_mean = v.tf_sum / len(tfidf_results) if tfidf_results else 0.0

            v.idf_min = min(r["idf"] for r in tfidf_results)
            v.idf_max = max(r["idf"] for r in tfidf_results)
            v.idf_sum = sum(r["idf"] for r in tfidf_results)
            v.idf_mean = v.idf_sum / len(tfidf_results) if tfidf_results else 0.0

            v.tf_idf_min = min(r["tf_idf"] for r in tfidf_results)
            v.tf_idf_max = max(r["tf_idf"] for r in tfidf_results)
            v.tf_idf_sum = sum(r["tf_idf"] for r in tfidf_results)
            v.tf_idf_mean = v.tf_idf_sum / len(tfidf_results) if tfidf_results else 0.0
            
            v.tf_variance = sum((r["tf"] - v.tf_mean) ** 2 for r in tfidf_results) / len(tfidf_results) if tfidf_results else 0.0
            v.idf_variance = sum((r["idf"] - v.idf_mean) ** 2 for r in tfidf_results) / len(tfidf_results) if tfidf_results else 0.0 
            v.tf_idf_variance = sum((r["tf_idf"] - v.tf_idf_mean) ** 2 for r in tfidf_results) / len(tfidf_results) if tfidf_results else 0.0

            record.qdr = v
            
            records.append(record)
        
    return records

print("Fetching user activities...")

user_activities = []
for user in os.listdir("crawl"):
    user_path = os.path.join("crawl", user)
    if not os.path.isdir(user_path):
        continue
        
    for json_file in os.listdir(user_path):
        if not json_file.endswith('.json'):
            continue
            
        with open(os.path.join(user_path, json_file)) as f:
            data = json.load(f)
            if data["chosen_index"] == -1:
                continue
            ua = UserActivity(data)
            ua.issuer = user
            user_activities.append(ua)

fetch_torrent_infos(user_activities)

print("Processing data and building corpus...")

# Filter invalid user activities
user_activities = [ua for ua in user_activities 
                   if ua.chosen_result.torrent_info is not None 
                   and ua.query.strip() != "" 
                   and ua.timestamp != 0 
                   and ua.timestamp < time.time()]

# Filter out results whose torrent info is missing
for ua in user_activities:
    ua.results = [res for res in ua.results if res.torrent_info is not None]

# Build the corpus
unique_documents = {res.infohash: res for ua in user_activities for res in ua.results}.values()
unique_queries = {ua.query for ua in user_activities}
corpus = {doc.infohash: doc.torrent_info.title.lower() for doc in unique_documents}
doc_ids = list(corpus.keys())
tfidf = TFIDF(corpus)
bm25 = BM25Okapi([tokenize(doc) for doc in corpus.values()])
del unique_documents, corpus

# Create DataFrame with user activity data
user_activity_df = pd.DataFrame([
    {
        'user': ua.issuer,
        'timestamp': ua.timestamp,
        'query': ua.query,
        'results': ua.results,
        'chosen_result': ua.chosen_result.infohash
    }
    for ua in user_activities
])

################
# MAIN DATASET #
################

print("Generating main dataset...")

# Group by qid and aggregate, keeping only results and chosen_result counts
aggregated_df = user_activity_df.groupby('query').agg({
    'results': 'first',
    'chosen_result': lambda x: dict(pd.Series(x).value_counts()) # infohash -> count
}).reset_index()

# Compile list of records
records = compile_records(aggregated_df)
train_records, val_records, test_records = split_dataset_by_qids(records)

shutil.rmtree(EXPORT_PATH, ignore_errors=True)
os.makedirs(EXPORT_PATH, exist_ok=True)

with open(f"{EXPORT_PATH}/train.txt", "w") as f:
    f.writelines(str(record) + "\n" for record in train_records)

with open(f"{EXPORT_PATH}/vali.txt", "w") as f:
    f.writelines(str(record) + "\n" for record in val_records)  

with open(f"{EXPORT_PATH}/test.txt", "w") as f:
    f.writelines(str(record) + "\n" for record in test_records)
 
del train_records, val_records, test_records

# Normalize features for main dataset
print("Normalizing features for main dataset...")
os.system(f"python ./allRank/reproducibility/normalize_features.py --ds_path {EXPORT_PATH}/")


#########################
# USER-SPECIFIC DATASET #
#########################

print("Generating user-specific datasets...")

# Get top 10 users by number of queries
user_query_counts = user_activity_df.groupby('user')['query'].nunique()
top_users = user_query_counts.nlargest(10).index

# Create per-user datasets
for user in top_users:
    # Get qids for this user's queries
    individual_user_activity_df = user_activity_df[user_activity_df['user'] == user]
    # user_queries = individual_user_activity_df['query'].unique()
    # user_qids = [list(unique_queries).index(q) for q in user_queries]
    
    # Filter records for this user's queries
    user_records = compile_records(individual_user_activity_df)
    
    # Split dataset
    train_records, val_records, test_records = split_dataset_by_qids(user_records)
    
    # Create user directory
    user_dir = os.path.join(EXPORT_PATH, "by_user", str(user))
    os.makedirs(user_dir, exist_ok=True)
    
    # Write files
    with open(f"{user_dir}/train.txt", "w") as f:
        f.writelines(str(record) + "\n" for record in train_records)
        
    with open(f"{user_dir}/vali.txt", "w") as f:
        f.writelines(str(record) + "\n" for record in val_records)
        
    with open(f"{user_dir}/test.txt", "w") as f:
        f.writelines(str(record) + "\n" for record in test_records)

# Create temporary combined files for normalization
temp_dir = os.path.join(EXPORT_PATH, "by_user", "temp")
os.makedirs(temp_dir, exist_ok=True)

# Combine all user files
for file_type in ['train.txt', 'vali.txt', 'test.txt']:
    with open(os.path.join(temp_dir, file_type), 'w') as outfile:
        for user in top_users:
            user_dir = os.path.join(EXPORT_PATH, "by_user", str(user))
            with open(os.path.join(user_dir, file_type)) as infile:
                outfile.write(infile.read())

# Run normalization on combined files
os.system(f"python ./allRank/reproducibility/normalize_features.py --ds_path {temp_dir}/")

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

# Clean up temp directory
shutil.rmtree(temp_dir)
