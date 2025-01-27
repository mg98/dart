from rank_bm25 import BM25Okapi
import math
import pandas as pd
from baselines.panache import compute_hit_counts
from baselines.maay import MAAY
from baselines.dinx import compute_click_counts
from baselines.grank import precompute_grank_score_fn
from joblib import Parallel, delayed
from common import *

EXPORT_PATH = "./ltr_dataset"

def qid_key(ua: UserActivity):
    return (ua.query, ua.issuer, ua.timestamp)

def write_records(export_path: str, roles: dict[str, list[ClickThroughRecord]]):
    os.makedirs(export_path, exist_ok=True)
    for role, records in roles.items():
        with open(f"{export_path}/{role}.txt", "w") as f:
            f.writelines(str(record) + "\n" for record in records)

class LTRDatasetMaker:
    def __init__(self, 
                 activities: list[UserActivity], 
                 hit_counts=None,
                 maay=None,
                 click_counts=None,
                 grank=None):
        self.activities = activities
        self.qid_mappings = {qid_key(ua) for ua in self.activities}
        self.hit_counts = hit_counts
        self.maay = maay
        self.click_counts = click_counts
        self.grank = grank
    
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
            for qid, (query, user, timestamp) in enumerate(self.qid_mappings):
                f.write(f"qid:{qid}\t{query}\t{user}\t{timestamp}\n")

    def compile_records(self) -> list[ClickThroughRecord]:

        def process_row(ua: UserActivity):
            qid = self.qid_mappings.index((ua.query, ua.issuer, ua.timestamp))
            query_terms = tokenize(ua.query)
            doc_indices = [self.doc_ids.index(result.infohash) for result in ua.results]
            bm25_scores = self.bm25.get_batch_scores(query_terms, doc_indices)
            
            row_records = []
            for i, result in enumerate(ua.results):
                record = ClickThroughRecord()
                record.qid = qid
                record.rel = int(result.infohash == ua.chosen_result.infohash)

                v = QueryDocumentRelationVector()
                v.seeders = result.seeders
                v.leechers = result.leechers
                v.age = ua.timestamp - result.torrent_info.timestamp
                v.bm25 = bm25_scores[i]


                other_activities = [
                    act for act in self.activities 
                    if not (
                        act.query == ua.query and 
                        act.issuer == ua.issuer and 
                        act.timestamp == ua.timestamp
                    )
                ]
                hit_counts = self.hit_counts or compute_hit_counts(other_activities)
                maay = self.maay or MAAY(other_activities)
                click_counts = self.click_counts or compute_click_counts(other_activities)
                grank = self.grank or precompute_grank_score_fn(other_activities)
                v.query_hit_count = sum(hit_counts.get(term, {}).get(result.infohash, 0) for term in query_terms)
                v.sp = maay.SP(result.infohash, ua.query)
                v.rel = maay.REL(result.infohash, ua.query)
                v.pop = maay.POP(result.infohash, ua.query)
                v.matching_score = maay.matching_score(ua.issuer, result.infohash)
                v.click_count = click_counts.get(result.infohash, 0)
                v.grank_score = grank(result.infohash, ua.issuer)

                v.pos = result.pos
                v.tag_count = len(result.torrent_info.tags)
                v.size =result.torrent_info.size

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
                row_records.append(record)

            return row_records
        
        parallel_records = Parallel(n_jobs=8, batch_size=8)(
            delayed(process_row)(ua)
            for ua in self.activities
        )

        records = [record for batch in parallel_records for record in batch]
        
        return records
    