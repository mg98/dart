from rank_bm25 import BM25Okapi
import pandas as pd
from baselines.panache import compute_hit_counts
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
        self.qid_mappings = {ua.query for ua in self.activities}
        self._build_corpus()
        self.hit_counts = compute_hit_counts(self.activities)

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
            for qid, query in enumerate(self.qid_mappings):
                f.write(f"qid:{qid}\t{query}\n")

    def _build_corpus(self):
        unique_documents = {doc.infohash: doc for ua in self.activities for doc in ua.results}.values()
        corpus = {doc.infohash: doc.torrent_info.title.lower() for doc in unique_documents}
        self.doc_ids = list(corpus.keys())
        self.tfidf = TFIDF(corpus)
        self.bm25 = BM25Okapi([tokenize(doc) for doc in corpus.values()])

    def compile_records(self) -> list[ClickThroughRecord]:
        # Group by qid and aggregate, keeping only results and chosen_result counts
        aggregated_df = self.df.groupby('query', sort=False).agg({
            'results': 'first',
            'timestamp': 'first',
            'chosen_result': lambda x: dict(pd.Series(x).value_counts()) # infohash -> count
        }).reset_index()

        records = []

        for _, row in aggregated_df.iterrows():
            qid = list(self.qid_mappings).index(row['query'])
            query_terms = tokenize(row['query'])
            doc_indices = [self.doc_ids.index(result.infohash) for result in row['results']]
            bm25_scores = self.bm25.get_batch_scores(query_terms, doc_indices)

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
                v.query_hit_count = sum(self.hit_counts[k][result.infohash] for k in query_terms)

                # aggregate tf idf features over all query terms
                tfidf_results = [self.tfidf.get_tf_idf(result.infohash, term) for term in query_terms]

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
    