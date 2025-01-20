from rank_bm25 import BM25Okapi
import math
import pandas as pd
from baselines.panache import compute_hit_counts
from baselines.maay import MAAY
from baselines.dinx import compute_click_counts
from joblib import Parallel, delayed
from common import *

EXPORT_PATH = "./ltr_dataset"

def write_records(export_path: str, roles: dict[str, list[ClickThroughRecord]]):
    os.makedirs(export_path, exist_ok=True)
    for role, records in roles.items():
        with open(f"{export_path}/{role}.txt", "w") as f:
            f.writelines(str(record) + "\n" for record in records)

class LTRDatasetMaker:
    def __init__(self, activities: list[UserActivity]):

        self.activities = activities
        self.qid_mappings = {(ua.query, ua.issuer) for ua in self.activities}
        self._build_corpus()
        self.hit_counts = compute_hit_counts(self.activities)
        self.maay = MAAY(self.activities)
        self.click_counts = compute_click_counts(self.activities)

        # Create DataFrame with user activity data
        self.df = pd.DataFrame([
            {
                'user': ua.issuer,
                'timestamp': ua.timestamp,
                'query': ua.query,
                'results': ua.results,
                'chosen_result': ua.chosen_result.infohash
            }
            for ua in self.activities
        ])
    
    @property
    def qid_mappings(self):
        return self._qid_mappings

    @qid_mappings.setter
    def qid_mappings(self, value):
        """
        Sorted for consistent qid assignments in parallel processing.
        """
        self._qid_mappings = tuple(sorted(value))

    def generate(self, export_path: str, normalize: bool = True):
        records = self.compile_records()
        train_records, vali_records, test_records = split_dataset_by_qids(records)

        write_records(export_path, {
            "train": train_records, 
            "vali": vali_records, 
            "test": test_records
        })

        if normalize:
            normalize_features(export_path)

    def write_queries(self, export_path: str):
        with open(f"{export_path}/queries.tsv", "w") as f:
            for qid, (query, user) in enumerate(self.qid_mappings):
                f.write(f"qid:{qid}\t{query}\t{user}\n")

    def _build_corpus(self):
        unique_documents = {doc.infohash: doc for ua in self.activities for doc in ua.results}.values()
        corpus = {doc.infohash: doc.torrent_info.title.lower() for doc in unique_documents}
        self.doc_ids = list(corpus.keys())
        self.tfidf = TFIDF(corpus)
        self.bm25 = BM25Okapi([tokenize(doc) for doc in corpus.values()])

    def compile_records(self) -> list[ClickThroughRecord]:
        # Group by qid and aggregate, keeping only results and chosen_result counts
        aggregated_df = self.df.groupby(['query', 'user'], sort=False).agg({
            'results': 'first',
            'timestamp': 'first',
            'chosen_result': lambda x: dict(pd.Series(x).value_counts())
        }).reset_index()

        def process_row(row, shared_resources):
            # Unpack shared resources
            bm25, tfidf, hit_counts, maay, click_counts, doc_ids, qid_mappings = shared_resources
            
            qid = qid_mappings.index((row['query'], row['user']))
            query_terms = tokenize(row['query'])
            doc_indices = [doc_ids.index(result.infohash) for result in row['results']]
            
            bm25_scores = bm25.get_batch_scores(query_terms, doc_indices)
            
            row_records = []
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
                v.age = row['timestamp'] - result.torrent_info.timestamp
                v.bm25 = bm25_scores[i]
                v.query_hit_count = sum(hit_counts[term][result.infohash] for term in query_terms)
                
                v.sp = maay.SP(result.infohash, row['query'])
                v.rel = maay.REL(result.infohash, row['query'])
                v.pop = maay.POP(result.infohash, row['query'])
                v.matching_score = maay.matching_score(row['user'], result.infohash)

                v.click_count = click_counts[result.infohash]

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
                row_records.append(record)

            return row_records
        
        # Package shared resources once
        shared_resources = (
            self.bm25,
            self.tfidf,
            self.hit_counts,
            self.maay,
            self.click_counts,
            self.doc_ids,
            self.qid_mappings
        )
        
        parallel_records = Parallel(n_jobs=8, batch_size=8)(
            delayed(process_row)(row, shared_resources)
            for _, row in aggregated_df.iterrows()
        )

        records = [record for batch in parallel_records for record in batch]
        
        return records
    