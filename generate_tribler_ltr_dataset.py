import re
from rank_bm25 import BM25Okapi
import pickle
import numpy as np
import pandas as pd
import shutil
from collections import defaultdict
from tqdm import tqdm
from utils.common import *
from baselines.panache import compute_hit_counts
from utils.ltr_helper import *

np.random.seed(42)

EXPORT_PATH = "tribler_data"
shutil.rmtree(EXPORT_PATH, ignore_errors=True)
os.makedirs(EXPORT_PATH, exist_ok=True)

with open('user_activities.pkl', 'rb') as f:
    user_activities = pickle.load(f)
np.random.shuffle(user_activities)

split_idx = int(0.9 * len(user_activities))
clicklogs = user_activities[:split_idx]
activities = user_activities[split_idx:]

print("Storing test activities...")
with open('tribler_data/test_activities.pkl', 'wb') as f:
    pickle.dump(activities, f)


print("Building corpus...")
qid_mappings = {qid_key(ua) for ua in clicklogs} | {qid_key(ua) for ua in activities}
unique_documents = {doc.infohash: doc for ua in clicklogs + activities for doc in ua.results}.values()
corpus = {doc.infohash: doc.torrent_info.title.lower() for doc in unique_documents}
corpus = Corpus(corpus)

print("Generating train/vali dataset...")
ltrdm_clicklogs = LTRDatasetMaker(clicklogs)
ltrdm_clicklogs.corpus = corpus
ltrdm_clicklogs.qid_mappings = qid_mappings
records = ltrdm_clicklogs.compile_records()
train_records, vali_records, _ = split_dataset_by_qids(records, train_ratio=0.8, val_ratio=0.2)
del records

print("Generating test dataset...")
ltrdm_activities = LTRDatasetMaker(activities)
ltrdm_activities.corpus = corpus
ltrdm_activities.qid_mappings = qid_mappings
ltrdm_activities.hit_counts = compute_hit_counts(clicklogs)
ltrdm_activities.click_counts = compute_click_counts(clicklogs)
test_records = ltrdm_activities.compile_records()

print("Generating main dataset...")

dataset_path = 'tribler_data'
write_records(dataset_path, {
    "train": train_records,
    "vali": vali_records,
    "test": test_records
})
normalize_features(dataset_path)

print("Done")