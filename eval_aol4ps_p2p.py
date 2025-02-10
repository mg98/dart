import pickle
from baselines.ltr import ltr_rank
import numpy as np
from common import mean_ndcg, TermBasedMetrics, tokenize, TFIDF
from collections import defaultdict
from tqdm import tqdm
from typing import Dict
import json
from rank_bm25 import BM25Okapi
from joblib import Parallel, delayed
import pandas as pd
np.random.seed(42)

K_RANGE = [5, 10, 30, None]

class QueryDocumentRelationVector:
    title: TermBasedMetrics = TermBasedMetrics()
    url: TermBasedMetrics = TermBasedMetrics()

    @property
    def features(self):
        def get_metrics_features(metrics: TermBasedMetrics) -> list:
            return [
                metrics.bm25,
                metrics.tf_min,
                metrics.tf_max,
                metrics.tf_mean,
                metrics.tf_sum,
                metrics.tf_variance,
                metrics.idf_min,
                metrics.idf_max,
                metrics.idf_mean,
                metrics.idf_sum,
                metrics.idf_variance,
                metrics.tf_idf_min,
                metrics.tf_idf_max,
                metrics.tf_idf_mean,
                metrics.tf_idf_sum,
                metrics.tf_idf_variance,
                metrics.cos_sim,
                metrics.covered_query_term_number,
                metrics.covered_query_term_ratio,
                metrics.char_len,
                metrics.term_len,
                metrics.total_query_terms,
                metrics.exact_match,
                metrics.match_ratio
            ]
            
        return [
            *get_metrics_features(self.title),  # Title features
            *get_metrics_features(self.url),    # URL features
        ]

    def __str__(self):
        return ' '.join(f'{i}:{val}' for i, val in enumerate(self.features))
    
class Corpus:
    def __init__(self, corpus: Dict[str, str]):
        self.corpus = corpus
        self.tfidf = TFIDF(corpus)
        self.bm25 = BM25Okapi([tokenize(t) for t in corpus.values()])

    def compute_features(self, doc_id: str, query_terms: list[str]) -> TermBasedMetrics:
        v = TermBasedMetrics()

        doc_index = list(self.corpus.keys()).index(doc_id)
        v.bm25 = self.bm25.get_batch_scores(query_terms, [doc_index])[0]

        tfidf_results = [self.tfidf.get_tf_idf(doc_id, term) for term in query_terms]

        v.tf_min = min(r["tf"] for r in tfidf_results) if tfidf_results else 0.0
        v.tf_max = max(r["tf"] for r in tfidf_results) if tfidf_results else 0.0
        v.tf_sum = sum(r["tf"] for r in tfidf_results)
        v.tf_mean = v.tf_sum / len(tfidf_results) if tfidf_results else 0.0

        v.idf_min = min(r["idf"] for r in tfidf_results) if tfidf_results else 0.0
        v.idf_max = max(r["idf"] for r in tfidf_results) if tfidf_results else 0.0
        v.idf_sum = sum(r["idf"] for r in tfidf_results)
        v.idf_mean = v.idf_sum / len(tfidf_results) if tfidf_results else 0.0

        v.tf_idf_min = min(r["tf_idf"] for r in tfidf_results) if tfidf_results else 0.0
        v.tf_idf_max = max(r["tf_idf"] for r in tfidf_results) if tfidf_results else 0.0
        v.tf_idf_sum = sum(r["tf_idf"] for r in tfidf_results)
        v.tf_idf_mean = v.tf_idf_sum / len(tfidf_results) if tfidf_results else 0.0
        
        v.tf_variance = sum((r["tf"] - v.tf_mean) ** 2 for r in tfidf_results) / len(tfidf_results) if tfidf_results else 0.0
        v.idf_variance = sum((r["idf"] - v.idf_mean) ** 2 for r in tfidf_results) / len(tfidf_results) if tfidf_results else 0.0 
        v.tf_idf_variance = sum((r["tf_idf"] - v.tf_idf_mean) ** 2 for r in tfidf_results) / len(tfidf_results) if tfidf_results else 0.0

        v.cos_sim = self.tfidf.get_cos_sim(doc_id, query_terms)

        v.covered_query_term_number = sum(1 for r in tfidf_results if r["tf"] > 0)
        v.covered_query_term_ratio = v.covered_query_term_number / len(query_terms)

        # Get document text from tfidf to calculate lengths
        doc_text = self.corpus[doc_id]
        v.char_len = len(doc_text)
        v.term_len = len(tokenize(doc_text))
        
        # Boolean features
        document_terms = tokenize(doc_text)
        matched_terms = set(query_terms) & set(document_terms)
        match_count = len(matched_terms)
        v.total_query_terms = len(query_terms)
        v.exact_match = 1 if match_count == v.total_query_terms else 0
        v.match_ratio = match_count / v.total_query_terms if v.total_query_terms > 0 else 0

        return v

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv('AOL4PS/data.csv', sep='\t')
    queries_df = pd.read_csv('AOL4PS/query.csv', sep='\t', index_col=1)
    docs_df = pd.read_csv('AOL4PS/doc.csv', sep='\t', index_col=1)

    # Add user filtering
    print("Filtering users...")
    user_counts = df['AnonID'].value_counts()
    eligible_users = user_counts[user_counts >= 10].index
    sampled_users = np.random.choice(eligible_users, size=min(1000, len(eligible_users)), replace=False)
    df = df[df['AnonID'].isin(sampled_users)]

    print("Generating clicklogs...")
    clicklog_data = []
    for i, (_, meta) in tqdm(enumerate(df.iterrows()), total=len(df)):
        candidates = meta['CandiList'].split('\t')
        for pos, candidate in enumerate(candidates):
            clicklog_data.append({
                'user_id': meta['AnonID'],
                'query': queries_df.loc[meta['QueryIndex']]['Query'],
                'doc_id': candidate,
                'document': docs_df.loc[candidate],
                'rel': int(pos == meta['ClickPos'])
            })

    clicklogs_df = pd.DataFrame(clicklog_data)

    # Group clicklogs by user
    print("Grouping clicklogs by user...")
    grouped_activities = defaultdict(list)
    for user_id, user_df in clicklogs_df.groupby('user_id'):
        doc_ids = list(user_df['doc_id'].unique())
        title_corpus = Corpus({doc_id: docs_df.loc[doc_id]['Title'] for doc_id in doc_ids})
        url_corpus = Corpus({doc_id: docs_df.loc[doc_id]['Url'] for doc_id in doc_ids})


    # Evaluate all activities for reference
    all_clicklogs = []
    for issuer, activities in tqdm(grouped_activities.items()):
        split_idx = int(len(activities) * 0.9)
        all_clicklogs.extend(activities[:split_idx])
    
    def process_user(issuer_activities):
        issuer, activities = issuer_activities
        split_idx = int(len(activities) * 0.9)
        reranked_activities = ltr_rank(activities[:split_idx], activities[split_idx:])
        reranked_activities_gossip = ltr_rank(all_clicklogs, activities[split_idx:])
            
        user_result = {
            "num_clicklogs": len(activities),
            "split_idx": split_idx,
            "local_ndcg": {},
            "gossip_ndcg": {},
        }

        for k in K_RANGE:
            user_ndcgs = mean_ndcg(reranked_activities, k=k)
            user_result["local_ndcg"][k] = float(np.mean(user_ndcgs))
            user_ndcgs = mean_ndcg(reranked_activities_gossip, k=k)
            user_result["gossip_ndcg"][k] = float(np.mean(user_ndcgs))
            print(f'User {issuer} ({len(activities)}) nDCG@{k}: {user_result["local_ndcg"][k]} (local), {user_result["gossip_ndcg"][k]} (gossip)')
            

    results = Parallel(n_jobs=4)(
        delayed(process_user)(item) for item in tqdm(grouped_activities.items())
    )

    # Save results to JSON file
    with open('results/aol4ps_ltr.json', 'w') as f:
        json.dump(results, f)
