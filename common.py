import sqlite3
import os
from copy import deepcopy
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict
from dataclasses import dataclass
import numpy as np
import pandas as pd
import functools
import time
import re
from rank_bm25 import BM25Okapi
from contextlib import contextmanager
from sklearn.metrics import ndcg_score
from sklearn.datasets import load_svmlight_file, dump_svmlight_file

# np.random.seed(42)

def ranking_func(_func=None, *, shuffle=True):
    def _decorate(func):
        @functools.wraps(func)
        def wrapper(arg1, arg2=None):
            if func.__name__ == 'tribler_rank':
                return func(arg1, arg2)
                
            clicklogs = arg1
            activities = deepcopy(arg2) if arg2 is not None else deepcopy(arg1)

            if shuffle:
                for ua in clicklogs:
                    np.random.shuffle(ua.results)
                for ua in activities:
                    np.random.shuffle(ua.results)
                np.random.shuffle(clicklogs)
                np.random.shuffle(activities)

            return func(clicklogs, activities)
        return wrapper
    
    if _func is not None and callable(_func):
        return _decorate(_func)
    
    return _decorate

def tokenize(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower().split()

class TorrentInfo:
    title: str
    tags: list[str]
    timestamp: float
    size: int

    def __init__(self, **kwargs):
        self.title = kwargs.get('title', '')
        self.tags = kwargs.get('tags', [])
        self.timestamp = kwargs.get('timestamp', 0)
        self.size = kwargs.get('size', 0)
    
    def __repr__(self):
        return f"TorrentInfo(title='{self.title}', tags={self.tags}, timestamp={self.timestamp}, size={self.size})"

    def __getstate__(self):
        return {
            'title': self.title,
            'tags': self.tags,
            'timestamp': self.timestamp,
            'size': self.size
        }

    def __setstate__(self, state):
        self.title = state['title']
        self.tags = state['tags'] 
        self.timestamp = state['timestamp']
        self.size = state['size']

class UserActivityTorrent:
    infohash: str
    seeders: int
    leechers: int
    pos: int
    torrent_info: TorrentInfo

    def __init__(self, data):
        self.infohash = data['infohash']
        self.seeders = data['seeders'] 
        self.leechers = data['leechers']
        self.torrent_info = None

    def __str__(self):
        return f"Infohash: {self.infohash}, Pos: {self.pos}, Seeders: {self.seeders}, Leechers: {self.leechers}, Torrent Info: {self.torrent_info}"
    
    def __getstate__(self):
        state = self.__dict__.copy()
        if isinstance(self.torrent_info, TorrentInfo):
            state['torrent_info'] = {
                'title': self.torrent_info.title,
                'tags': self.torrent_info.tags,
                'timestamp': self.torrent_info.timestamp,
                'size': self.torrent_info.size,
            }
        return state

    def __setstate__(self, state):
        if isinstance(state, UserActivityTorrent):
            self.infohash = state.infohash
            self.seeders = state.seeders
            self.leechers = state.leechers
            self.pos = state.pos
            self.torrent_info = state.torrent_info
        else:
            self.__dict__.update(state)
            if isinstance(self.torrent_info, dict):
                self.torrent_info = TorrentInfo(**self.torrent_info)

class UserActivity:
    issuer: str
    query: str
    timestamp: int
    results: list[UserActivityTorrent]
    chosen_result: UserActivityTorrent

    def __init__(self, data: dict):
        self.issuer = data['issuer']
        self.query = data['query']
        self.timestamp = int(data['timestamp'] / 1000)
        self.results = []
        for pos, result in enumerate(data['results']):
            torrent = UserActivityTorrent(result)
            torrent.pos = pos
            self.results.append(torrent)
        self.chosen_result = self.results[data['chosen_index']]

    @property
    def chosen_index(self) -> int:
        """Returns the index of the chosen result in the results list"""
        for i, result in enumerate(self.results):
            if result.infohash == self.chosen_result.infohash:
                return i
        return -1

    def __repr__(self):
        return (f"UserActivity(issuer={self.issuer}, query={self.query}, "
                f"timestamp={self.timestamp}, chosen_result={self.chosen_result}, results=[{len(self.results)}  items...])")

    def __getstate__(self):
        state = self.__dict__.copy()
        state['results'] = [result.__getstate__() for result in self.results]
        return state

    def __setstate__(self, state):
        # Extract the list of results from the state
        results_state = state.pop('results', [])
        chosen_result_state = state.pop('chosen_result', None)  # Extract chosen_result separately

        # Update the rest of the fields
        self.__dict__.update(state)

        # Convert each of the dicts in results_state back into a UserActivityTorrent
        self.results = []
        for r_state in results_state:
            torrent = UserActivityTorrent.__new__(UserActivityTorrent)
            torrent.__setstate__(r_state)
            self.results.append(torrent)

        # Handle chosen_result reconstruction
        if chosen_result_state is None:
            self.chosen_result = None
        else:
            # Create a new UserActivityTorrent for chosen_result
            chosen_torrent = UserActivityTorrent.__new__(UserActivityTorrent)
            chosen_torrent.__setstate__(chosen_result_state)
            
            # Find the matching torrent in results
            self.chosen_result = next(
                (t for t in self.results if t.infohash == chosen_torrent.infohash), 
                None
            )
            
            if self.chosen_result is None:
                print(f"Warning: Could not find matching torrent for chosen_result with infohash: {chosen_torrent.infohash}")


def fetch_torrent_infos(user_activities: list[UserActivity]):
    """Fetch torrent info for a list of UserActivityTorrent objects using batched SQL queries"""
    all_torrents = [t for ua in user_activities for t in ua.results]
    infohashes = list(set(t.infohash for t in all_torrents))
    
    BATCH_SIZE = 50000
    torrent_info_map = {}
    
    conn = sqlite3.connect(os.path.expanduser('./metadata.db'))
    cursor = conn.cursor()

    for i in range(0, len(infohashes), BATCH_SIZE):
        batch = infohashes[i:i + BATCH_SIZE]
        placeholders = ','.join(['?' for _ in batch])
        
        cursor.execute(f"""
            SELECT infohash_hex, title, tags, timestamp/1000 as timestamp, size 
            FROM ChannelNode
            WHERE infohash_hex IN ({placeholders})
            """, batch)
        
        results = cursor.fetchall()
        
        for result in results:
            info = TorrentInfo()
            info.title = result[1]
            info.tags = result[2].split(',') if result[2] else []
            info.timestamp = result[3]
            info.size = result[4]
            torrent_info_map[result[0]] = info
    
    conn.close()
    
    # Update torrents in original user_activities
    found = 0
    not_found = 0
    for ua in user_activities:
        for torrent in ua.results:
            if torrent.infohash in torrent_info_map:
                torrent.torrent_info = torrent_info_map[torrent.infohash]
                found += 1

                if ua.chosen_result.infohash == torrent.infohash:
                    ua.chosen_result = torrent
            else:
                not_found += 1
    
    print(f'Found {found} torrents, skipped {not_found}')

    

class TFIDF:
    def __init__(self, corpus: Dict[str, str]):
        self.documents = list(corpus.values())
        self.doc_ids = list(corpus.keys())
        self.vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        # Precompute term counts for each document
        self.term_counts = [doc.split() for doc in self.documents]
        self.total_terms = [len(doc) for doc in self.term_counts]

    def get_tf_idf(self, doc_id: str, term: str) -> dict[str, float]:
        try:
            word_idx = list(self.feature_names).index(term)
        except ValueError:
            return { "tf": 0, "tf_idf": 0, "idf": 0 }
    
        doc_idx = self.doc_ids.index(doc_id)
        tf_idf = self.tfidf_matrix[doc_idx, word_idx]
        idf = self.vectorizer.idf_[word_idx]
        tf = tf_idf / idf if idf != 0 else 0
        
        return { "tf": tf, "tf_idf": tf_idf, "idf": idf }
    
@dataclass
class QueryDocumentRelationVector:
    tf_min: float = 0.0
    tf_max: float = 0.0
    tf_mean: float = 0.0
    tf_sum: float = 0.0
    tf_variance: float = 0.0
    idf_min: float = 0.0
    idf_max: float = 0.0
    idf_mean: float = 0.0
    idf_sum: float = 0.0
    idf_variance: float = 0.0
    tf_idf_min: float = 0.0
    tf_idf_max: float = 0.0
    tf_idf_mean: float = 0.0
    tf_idf_sum: float = 0.0
    tf_idf_variance: float = 0.0
    bm25: float = 0.0
    seeders: int = 0
    leechers: int = 0
    age: float = 0.0
    query_hit_count: int = 0
    sp: float = 0.0
    rel: float = 0.0
    pop: float = 0.0
    matching_score: float = 0.0
    click_count: float = 0.0
    grank_score: float = 0.0
    pos: int = 0
    tag_count: int = 0
    size: int = 0

    @property
    def features(self):
        # There is a bug where when the last feature value is 0, sklearn trims it from the dataset,
        # which yields inconsistent shapes and crashes the training. As a dirty fix, I put the age
        # feature at last position.
        return [self.seeders, self.leechers, self.bm25,
                 self.tf_min, self.tf_max, self.tf_mean, self.tf_sum,
                 self.idf_min, self.idf_max, self.idf_mean, self.idf_sum,
                 self.tf_idf_min, self.tf_idf_max, self.tf_idf_mean, self.tf_idf_sum,
                 self.tf_variance, self.idf_variance, self.tf_idf_variance, 
                 self.sp, self.rel, self.pop, self.matching_score, 
                 self.click_count, self.grank_score, self.query_hit_count,
                #  self.pos, 
                 self.tag_count, 
                #  self.size,
                 self.age]

    def __str__(self):
        return ' '.join(f'{i}:{val}' for i, val in enumerate(self.features))

class ClickThroughRecord:
    rel: float
    qid: int
    qdr: QueryDocumentRelationVector

    def __init__(self, rel=0.0, qid=0, qdr=None): 
        self.rel = rel
        self.qid = qid
        self.qdr = qdr

    def to_dict(self):
        return {
            'rel': self.rel,
            'qid': self.qid,
            'qdr': self.qdr
        }

    def __str__(self):
        return f'{self.rel} qid:{self.qid} {self.qdr}'
    
def split_dataset_by_qids(records, train_ratio=0.8, val_ratio=0.1):
    """
    Split records into train/validation/test sets based on query IDs.
    
    Args:
        records: list containing the records
        train_ratio: Proportion of data for training (default 0.8)
        val_ratio: Proportion of data for validation (default 0.1)
        
    Returns:
        tuple of (train_records, val_records, test_records) as lists of ClickThroughRecord objects
    """
    records_df = pd.DataFrame([record.to_dict() for record in records])
    qids = records_df['qid'].unique()
    # np.random.shuffle(qids) # For some reason, shuffling the qids leads to worse results
    
    # Calculate split sizes
    n_qids = len(qids)
    train_size = int(train_ratio * n_qids)
    val_size = int(val_ratio * n_qids)
    
    # Split qids into train/val/test
    train_qids = qids[:train_size]
    val_qids = qids[train_size:train_size+val_size]
    test_qids = qids[train_size+val_size:]
    
    # Filter records by qid
    train_records_df = records_df[records_df['qid'].isin(train_qids)]
    val_records_df = records_df[records_df['qid'].isin(val_qids)]
    test_records_df = records_df[records_df['qid'].isin(test_qids)]
    
    # Convert to ClickThroughRecord objects
    train_records = [ClickThroughRecord(**record) for _, record in train_records_df.iterrows()]
    val_records = [ClickThroughRecord(**record) for _, record in val_records_df.iterrows()]
    test_records = [ClickThroughRecord(**record) for _, record in test_records_df.iterrows()]
    
    return train_records, val_records, test_records


FEATURES_WITHOUT_LOGARITHM = [
    5, 6, 7, 8, 9, 15, 19, 57, 58, 62, 75, 79, 85, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 121, 122, 127, 129, 130]
FEATURES_NEGATIVE = [110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 123, 124]

def normalize_features(ds_path: str, 
                       features_without_logarithm: list[int] = FEATURES_WITHOUT_LOGARITHM, 
                       features_negative: list[int] = FEATURES_NEGATIVE):
    """
    Normalize features in the dataset.
    Adapted from https://github.com/allegro/allRank/blob/master/reproducibility/normalize_features.py.
    """
    
    x_train, y_train, query_ids_train = load_svmlight_file(os.path.join(ds_path, "train.txt"), query_id=True)
    x_test, y_test, query_ids_test = load_svmlight_file(os.path.join(ds_path, "test.txt"), query_id=True)
    x_vali, y_vali, query_ids_vali = load_svmlight_file(os.path.join(ds_path, "vali.txt"), query_id=True)

    x_train_transposed = x_train.toarray().T
    x_test_transposed = x_test.toarray().T
    x_vali_transposed = x_vali.toarray().T

    x_train_normalized = np.zeros_like(x_train_transposed)
    x_test_normalized = np.zeros_like(x_test_transposed)
    x_vali_normalized = np.zeros_like(x_vali_transposed)

    eps_log = 1e-2
    eps = 1e-6

    for i, feat in enumerate(x_train_transposed):
        feature_vector_train = feat
        feature_vector_test = x_test_transposed[i, ]
        feature_vector_vali = x_vali_transposed[i, ]

        if i in features_negative:
            feature_vector_train = (-1) * feature_vector_train
            feature_vector_test = (-1) * feature_vector_test
            feature_vector_vali = (-1) * feature_vector_vali

        if i not in features_without_logarithm:
            # log only if all values >= 0
            if np.all(feature_vector_train >= 0) & np.all(feature_vector_test >= 0) & np.all(feature_vector_vali >= 0):
                feature_vector_train = np.log(feature_vector_train + eps_log)
                feature_vector_test = np.log(feature_vector_test + eps_log)
                feature_vector_vali = np.log(feature_vector_vali + eps_log)
            else:
                print("Some values of feature no. {} are still < 0 which is why the feature won't be normalized".format(i))

        mean = np.mean(feature_vector_train)
        std = np.std(feature_vector_train)
        feature_vector_train = (feature_vector_train - mean) / (std + eps)
        feature_vector_test = (feature_vector_test - mean) / (std + eps)
        feature_vector_vali = (feature_vector_vali - mean) / (std + eps)
        x_train_normalized[i, ] = feature_vector_train
        x_test_normalized[i, ] = feature_vector_test
        x_vali_normalized[i, ] = feature_vector_vali

    ds_normalized_path = os.path.join(ds_path, "_normalized")
    os.makedirs(ds_normalized_path, exist_ok=True)

    train_normalized_path = os.path.join(ds_normalized_path, "train.txt")
    with open(train_normalized_path, "w"):
        dump_svmlight_file(x_train_normalized.T, y_train, train_normalized_path, query_id=query_ids_train)

    test_normalized_path = os.path.join(ds_normalized_path, "test.txt")
    with open(test_normalized_path, "w"):
        dump_svmlight_file(x_test_normalized.T, y_test, test_normalized_path, query_id=query_ids_test)

    vali_normalized_path = os.path.join(ds_normalized_path, "vali.txt")
    with open(vali_normalized_path, "w"):
        dump_svmlight_file(x_vali_normalized.T, y_vali, vali_normalized_path, query_id=query_ids_vali)

def calc_mrr(ua):
    """Calculate Mean Reciprocal Rank for a single user activity"""
    for i, res in enumerate(ua.results):
        if res.infohash == ua.chosen_result.infohash:
            return 1.0 / (i + 1)
    return 0.0

def mean_mrr(user_activities: list[UserActivity]) -> float:
    return np.mean([calc_mrr(ua) for ua in user_activities])

def calc_ndcg(ua: UserActivity, k=None) -> float:
    """Calculate nDCG@k for a single user activity
    Args:
        ua: user activity
        k: number of top results to consider. If None, considers all results
    """
    true_relevance = [1 if res.infohash == ua.chosen_result.infohash else 0 for res in ua.results]
    predicted_relevance = [1/np.log2(i+2) for i in range(len(ua.results))]
    
    if k is not None:
        # Truncate both lists to k elements
        true_relevance = true_relevance[:k]
        predicted_relevance = predicted_relevance[:k]
    
    return ndcg_score([true_relevance], [predicted_relevance])

def mean_ndcg(user_activities: list[UserActivity], k=None) -> float:
    return np.round(np.mean([calc_ndcg(ua, k) for ua in user_activities]), 3)

@contextmanager
def timing():
    """Context manager that measures execution time and formats output as XhYmZs"""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        print(f"Time taken: {hours}h{minutes}m{seconds}s")


def build_corpus(activities: list[UserActivity]):
    unique_documents = {doc.infohash: doc for ua in activities for doc in ua.results}.values()
    corpus = {doc.infohash: doc.torrent_info.title.lower() for doc in unique_documents}
    return {
        'doc_ids': list(corpus.keys()),
        'tfidf': TFIDF(corpus),
        'bm25': BM25Okapi([tokenize(doc) for doc in corpus.values()])
    }