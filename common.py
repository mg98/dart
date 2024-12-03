import sqlite3
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict
from dataclasses import dataclass

class TorrentInfo:
    title: str
    tags: list[str]
    timestamp: float
    size: int

class UserActivityTorrent:
    infohash: str
    seeders: int
    leechers: int
    torrent_info: TorrentInfo

    def __init__(self, data):
        self.infohash = data['infohash']
        self.seeders = data['seeders'] 
        self.leechers = data['leechers']
        self.torrent_info = None

    def __str__(self):
        return f"Infohash: {self.infohash}, Seeders: {self.seeders}, Leechers: {self.leechers}, Torrent Info: {self.torrent_info}"

class UserActivity:
    issuer: str
    query: str
    chosen_index: int
    timestamp: int
    results: list[UserActivityTorrent]

    def __init__(self, data: dict):
        self.query = data['query']
        self.chosen_index = data['chosen_index']
        self.timestamp = int(data['timestamp'] / 1000)
        self.results = []
        for result in data['results']:
            torrent = UserActivityTorrent(result)
            self.results.append(torrent)

    @property
    def chosen_result(self) -> UserActivityTorrent:
        return self.results[self.chosen_index]

    def __str__(self):
        return f"Issuer: {self.issuer}, Query: {self.query}, Chosen Index: {self.chosen_index}, Timestamp: {self.timestamp}, Results: {self.results}"

def fetch_torrent_infos(user_activities: list[UserActivity]):
    """Fetch torrent info for a list of UserActivityTorrent objects using a single SQL query"""
    all_torrents = [t for ua in user_activities for t in ua.results]
    
    infohashes = [t.infohash for t in all_torrents]
    placeholders = ','.join(['?' for _ in infohashes])

    conn = sqlite3.connect(os.path.expanduser('./metadata.db'))
    cursor = conn.cursor()
    
    cursor.execute(f"""
        SELECT infohash_hex, title, tags, timestamp/1000 as timestamp, size 
        FROM ChannelNode
        WHERE infohash_hex IN ({placeholders})
        """, infohashes)
    
    results = cursor.fetchall()
    conn.close()

    torrent_info_map = {}
    for result in results:
        info = TorrentInfo()
        info.title = result[1]
        info.tags = result[2].split(',') if result[2] else []
        info.timestamp = result[3]
        info.size = result[4]
        torrent_info_map[result[0]] = info
    
    # Update torrents in original user_activities
    for ua in user_activities:
        for torrent in ua.results:
            if torrent.infohash in torrent_info_map:
                torrent.torrent_info = torrent_info_map[torrent.infohash]
            else:
                print(f'torrent infohash {torrent.infohash} not found')

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

    def __str__(self):
        fields = [self.seeders, self.leechers, self.age, self.bm25, 
                 self.tf_min, self.tf_max, self.tf_mean, self.tf_sum,
                 self.idf_min, self.idf_max, self.idf_mean, self.idf_sum,
                 self.tf_idf_min, self.tf_idf_max, self.tf_idf_mean, self.tf_idf_sum,
                 self.tf_variance, self.idf_variance, self.tf_idf_variance]
        return ' '.join(f'{i+1}:{val}' for i, val in enumerate(fields))

class ClickThroughRecord:
    rel: bool
    qid: int
    qdr: QueryDocumentRelationVector

    def __str__(self):
        return f'{int(self.rel)} qid:{self.qid} {self.qdr}'