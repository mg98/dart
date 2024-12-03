import json
import re
from rank_bm25 import BM25Okapi
import random
from common import *

def tokenize(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower().split()

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
            ua = UserActivity(data)
            ua.issuer = user
            user_activities.append(ua)

fetch_torrent_infos(user_activities)

# filter out results whose torrent info could not be fetched
for ua in user_activities:
    ua.results = list(filter(lambda res: res.torrent_info is not None, ua.results))

unique_documents = {res.infohash: res for ua in user_activities for res in ua.results}.values()
unique_queries = {ua.query for ua in user_activities}
corpus = {doc.infohash: doc.torrent_info.title.lower() for doc in unique_documents}
doc_ids = list(corpus.keys())
tfidf = TFIDF(corpus)
bm25 = BM25Okapi([tokenize(doc) for doc in corpus.values()])
del unique_documents, corpus

output_path = "tribler_ltr_dataset.txt"
try: os.remove(output_path)
except FileNotFoundError: pass

for ua in user_activities:
    query_terms = tokenize(ua.query)
    doc_indices = [doc_ids.index(result.infohash) for result in ua.results]
    bm25_scores = bm25.get_batch_scores(query_terms, doc_indices)

    for i, result in enumerate(ua.results):

        v = QueryDocumentRelationVector()
        v.seeders = result.seeders
        v.leechers = result.leechers
        v.age = ua.timestamp - result.torrent_info.timestamp
        v.bm25 = bm25_scores[i]

        # aggregate tf idf features over all query terms
        tfidf_results = [tfidf.get_tf_idf(result.infohash, term) for term in query_terms]
        
        v.tf_min = min(r["tf"] for r in tfidf_results)
        v.tf_max = max(r["tf"] for r in tfidf_results)
        v.tf_mean = v.tf_sum / len(tfidf_results) if tfidf_results else 0.0
        v.tf_sum = sum(r["tf"] for r in tfidf_results)

        v.idf_min = min(r["idf"] for r in tfidf_results)
        v.idf_max = max(r["idf"] for r in tfidf_results)
        v.idf_mean = v.idf_sum / len(tfidf_results) if tfidf_results else 0.0
        v.idf_sum = sum(r["idf"] for r in tfidf_results)

        v.tf_idf_min = min(r["tf_idf"] for r in tfidf_results)
        v.tf_idf_max = max(r["tf_idf"] for r in tfidf_results)
        v.tf_idf_mean = v.tf_idf_sum / len(tfidf_results) if tfidf_results else 0.0
        v.tf_idf_sum = sum(r["tf_idf"] for r in tfidf_results)
        
        v.tf_variance = sum((r["tf"] - v.tf_mean) ** 2 for r in tfidf_results) / len(tfidf_results) if tfidf_results else 0.0
        v.idf_variance = sum((r["idf"] - v.idf_mean) ** 2 for r in tfidf_results) / len(tfidf_results) if tfidf_results else 0.0 
        v.tf_idf_variance = sum((r["tf_idf"] - v.tf_idf_mean) ** 2 for r in tfidf_results) / len(tfidf_results) if tfidf_results else 0.0

        ctr = ClickThroughRecord()
        ctr.rel = ua.chosen_index == i
        ctr.qid = list(unique_queries).index(ua.query)
        ctr.qdr = v

        print(ctr)
        with open(output_path, "a") as f:
            f.write(str(ctr) + "\n")