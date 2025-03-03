from collections import defaultdict
from joblib import Parallel, delayed
import numpy as np
from utils.common import ranking_func

@ranking_func
def maay_rank(clicklogs: list, activities: list = None):

    # Extract sets of words and documents
    W = set()
    D = set()
    for ua in clicklogs:
        query_words = ua.query.lower().split()
        W.update(query_words)
        if ua.chosen_result:
            D.add(ua.chosen_result.infohash)
    W = list(W)  # fix ordering
    D = list(D)

    # Precompute claims and votes
    claims = defaultdict(lambda: defaultdict(int))
    votes = defaultdict(lambda: defaultdict(int))

    # Also keep track of all issuers (b)
    nodes = set()

    for ua in clicklogs:
        b = ua.issuer
        nodes.add(b)
        query_words = ua.query.lower().split()
        # Update claims
        for w in query_words:
            claims[b][w] += 1
        # Update votes
        if ua.chosen_result:
            d = ua.chosen_result.infohash
            for w in query_words:
                votes[d][w] += 1

    # Precompute totals for faster lookups
    total_claims_per_b = {}
    for b in claims:
        total_claims_per_b[b] = sum(claims[b].values())

    total_votes_per_d = defaultdict(int)
    for d in votes:
        total_votes_per_d[d] = sum(votes[d].values())

    # votes_per_w: total votes for word w across all documents
    votes_per_w = defaultdict(int)
    for d in votes:
        for w, c in votes[d].items():
            votes_per_w[w] += c

    # Define SP, REL, POP using precomputed structures
    def SP(b, w):
        denom = total_claims_per_b[b]
        if denom == 0:
            return 0.0
        return claims[b][w] / denom

    def REL(d, w):
        sum_votes_d = total_votes_per_d[d]
        if sum_votes_d == 0:
            return 0.0
        return votes[d][w] / sum_votes_d

    def POP(d, w):
        total_w = votes_per_w[w]
        if total_w == 0:
            return 0.0
        return votes[d][w] / total_w

    def compute_matching_for_pair(z, d, W_list):
        s = 0.0
        # Compute sum over all words
        for w in W_list:
            s += REL(d, w) * SP(z, w)
        return (z, d, s)

    # Parallelize pre-computation of matching scores
    results = []
    for z in nodes:
        for d in D:
            results.append(compute_matching_for_pair(z, d, W))

    matching_scores = defaultdict(lambda: defaultdict(int))
    for z, d, s in results:
        matching_scores[z][d] = s

    def DRS(z, d, w):
        # DRS(z, d, w) = REL(d, w)*POP(d, w)*matching(z, d)
        return REL(d, w) * POP(d, w) * matching_scores[z][d]

    # Now re-rank results
    # For each user activity, we only sum over query words:
    # score = sum(DRS(ua.issuer, x.infohash, w) for w in ua.query.split())

    for ua in activities:
        z = ua.issuer
        query_words = ua.query.lower().split()

        # Precompute scores once per result
        scored_results = []
        for r in ua.results:
            d = r.infohash
            score = 0.0
            for w in query_words:
                score += DRS(z, d, w)
            scored_results.append((score, r))

        # Sort
        scored_results.sort(key=lambda x: x[0], reverse=True)
        ua.results = [r for (_, r) in scored_results]

    return activities


@ranking_func
def maay_rank_numpy(clicklogs: list, activities: list = None):

    # Extract sets of words and documents, and nodes
    W_set = set()
    D_set = set()
    nodes_set = set()

    for ua in clicklogs:
        query_words = ua.query.lower().split()
        W_set.update(query_words)
        if ua.chosen_result:
            D_set.add(ua.chosen_result.infohash)
        nodes_set.add(ua.issuer)

    W = list(W_set)
    D = list(D_set)
    nodes = list(nodes_set)

    w2i = {w: i for i, w in enumerate(W)}
    d2i = {d: i for i, d in enumerate(D)}
    b2i = {b: i for i, b in enumerate(nodes)}

    num_b = len(nodes)
    num_w = len(W)
    num_d = len(D)

    # Initialize arrays
    # claims_array[b,w]
    claims_array = np.zeros((num_b, num_w), dtype=float)
    # votes_array[d,w]
    votes_array = np.zeros((num_d, num_w), dtype=float)

    # Populate claims and votes
    for ua in clicklogs:
        b_idx = b2i[ua.issuer]
        query_words = ua.query.lower().split()
        for w in query_words:
            w_idx = w2i[w]
            claims_array[b_idx, w_idx] += 1
        if ua.chosen_result:
            d_idx = d2i[ua.chosen_result.infohash]
            for w in query_words:
                w_idx = w2i[w]
                votes_array[d_idx, w_idx] += 1

    # Compute totals
    # sum_claims_per_b[b] = sum of claims for node b over all w
    sum_claims_per_b = claims_array.sum(axis=1)  # shape (num_b,)
    # sum_votes_per_d[d] = sum of votes for doc d over all w
    sum_votes_per_d = votes_array.sum(axis=1)  # shape (num_d,)
    # votes_per_w[w] = sum of votes over all d
    votes_per_w = votes_array.sum(axis=0)  # shape (num_w,)

    # Compute SP, REL, POP as matrices
    # Avoid division by zero by using where= parameter in numpy.divide if needed
    with np.errstate(divide='ignore', invalid='ignore'):
        SP_matrix = np.divide(claims_array, sum_claims_per_b[:, None], where=(sum_claims_per_b[:,None]!=0))
        # If divide by zero occurs, result is NaN, replace with 0
        SP_matrix = np.nan_to_num(SP_matrix)

        REL_matrix = np.divide(votes_array, sum_votes_per_d[:, None], where=(sum_votes_per_d[:,None]!=0))
        REL_matrix = np.nan_to_num(REL_matrix)

        POP_matrix = np.divide(votes_array, votes_per_w[None, :], where=(votes_per_w[None,:]!=0))
        POP_matrix = np.nan_to_num(POP_matrix)

    # Compute matching_scores[b,d] = sum_w REL(d,w)*SP(b,w)
    # This is (b,w) @ (w,d) => (b,d)
    # We have SP_matrix(b,w) and REL_matrix(d,w). We want Σ_w SP(b,w)*REL(d,w)
    # So we do: matching_scores = SP_matrix @ REL_matrix.T
    matching_scores = SP_matrix @ REL_matrix.T  # shape (num_b, num_d)

    # We'll now need to rank results per user activity
    # DRS(z, d, w) = REL(d,w)*POP(d,w)*matching_scores[z,d]
    # For a given query, we must sum over w in the query:
    # score = Σ_w DRS(z,d,w) = Σ_w REL(d,w)*POP(d,w)*matching_scores[z,d]
    # matching_scores[z,d] does not depend on w, so:
    # score = matching_scores[z,d] * Σ_w [REL(d,w)*POP(d,w)]

    # Precompute REL*POP for convenience: (d,w)
    REL_POP = REL_matrix * POP_matrix  # shape (d,w)

    # Now for each user activity:
    for ua in activities:
        # Check if the user (issuer) is known
        if ua.issuer not in b2i:
            # Option 1: Skip ranking entirely. Leave original order.
            # continue

            # Option 2: Force them to bottom by giving a score of 0
            scored_results = [(0.0, r) for r in ua.results]
            scored_results.sort(key=lambda x: x[0], reverse=True)
            ua.results = [r for (_, r) in scored_results]
            continue

        z_idx = b2i[ua.issuer]
        query_words = ua.query.split()
        # Filter out words not in w2i
        qw_indices = [w2i[w] for w in query_words if w in w2i]

        # If there are no known words, skip or leave original order
        if not qw_indices:
            continue

        # Precompute sum_w for known words in query: (REL(d, w)*POP(d, w))
        rel_pop_sums_for_query = REL_POP[:, qw_indices].sum(axis=1)  # shape: (num_d,)

        # matching_scores for this user
        z_d_scores = matching_scores[z_idx, :]  # shape: (num_d,)

        scored_results = []
        for r in ua.results:
            # If r.infohash is unknown, treat it specially
            if r.infohash not in d2i:
                # Option 1: Keep them in current place
                # scored_results.append((0.0, r))

                # Option 2: Skip or force them to bottom
                scored_results.append((0.0, r))
                continue

            d_idx = d2i[r.infohash]
            # score = matching_scores[z_idx, d_idx] * Σ(REL(d,w)*POP(d,w)) for w in query
            score = z_d_scores[d_idx] * rel_pop_sums_for_query[d_idx]
            scored_results.append((score, r))

        # Sort results by score, descending
        scored_results.sort(key=lambda x: x[0], reverse=True)
        ua.results = [r for (_, r) in scored_results]

    return activities


class MAAY:
    def __init__(self, clicklogs):
        # Extract sets of words, documents, and nodes
        W_set = set()
        D_set = set()
        nodes_set = set()

        for ua in clicklogs:
            query_words = ua.query.lower().split()
            W_set.update(query_words)
            if ua.chosen_result:
                D_set.add(ua.chosen_result.infohash)
            nodes_set.add(ua.issuer)

        # Create ordered lists and mapping dictionaries
        self.W = list(W_set)
        self.D = list(D_set)
        self.nodes = list(nodes_set)
        
        self.w2i = {w: i for i, w in enumerate(self.W)}
        self.d2i = {d: i for i, d in enumerate(self.D)}
        self.b2i = {b: i for i, b in enumerate(self.nodes)}

        num_b = len(self.nodes)
        num_w = len(self.W)
        num_d = len(self.D)

        # Initialize and populate arrays
        claims_array = np.zeros((num_b, num_w), dtype=float)
        votes_array = np.zeros((num_d, num_w), dtype=float)

        for ua in clicklogs:
            b_idx = self.b2i[ua.issuer]
            query_words = ua.query.lower().split()
            for w in query_words:
                w_idx = self.w2i[w]
                claims_array[b_idx, w_idx] += 1
            if ua.chosen_result:
                d_idx = self.d2i[ua.chosen_result.infohash]
                for w in query_words:
                    w_idx = self.w2i[w]
                    votes_array[d_idx, w_idx] += 1

        # Compute totals
        sum_claims_per_b = claims_array.sum(axis=1)
        sum_votes_per_d = votes_array.sum(axis=1)
        votes_per_w = votes_array.sum(axis=0)

        # Compute matrices
        with np.errstate(divide='ignore', invalid='ignore'):
            self.SP_matrix = np.divide(claims_array, sum_claims_per_b[:, None], 
                                     where=(sum_claims_per_b[:,None]!=0))
            self.SP_matrix = np.nan_to_num(self.SP_matrix)

            self.REL_matrix = np.divide(votes_array, sum_votes_per_d[:, None], 
                                      where=(sum_votes_per_d[:,None]!=0))
            self.REL_matrix = np.nan_to_num(self.REL_matrix)

            self.POP_matrix = np.divide(votes_array, votes_per_w[None, :], 
                                      where=(votes_per_w[None,:]!=0))
            self.POP_matrix = np.nan_to_num(self.POP_matrix)

        # Compute matching scores
        self.matching_scores = self.SP_matrix @ self.REL_matrix.T

    def SP(self, b, query):
        """
        Get aggregated specificity of all words in a query for user b.
        Specificity measures how characteristic query words are for this user's search behavior.
        
        Args:
            b: user/issuer
            query: search query string
        Returns:
            float: average specificity across all known words in query
        """
        if b not in self.b2i:
            return 0.0
        
        words = query.lower().split()
        known_words = [w for w in words if w in self.w2i]
        if not known_words:
            return 0.0
        
        specs = [self.SP_matrix[self.b2i[b], self.w2i[w]] for w in known_words]
        return sum(specs) / len(known_words)

    def REL(self, d, query):
        """
        Get aggregated relevance of all query words for document identified by infohash d.
        Relevance is measured by how often the words appear in successful queries for this document.
        
        Args:
            d: document infohash
            query: search query string
        Returns:
            float: average relevance score across all known words in query
        """
        if d not in self.d2i:
            return 0.0
        
        words = query.lower().split()
        known_words = [w for w in words if w in self.w2i]
        if not known_words:
            return 0.0
        
        relevances = [self.REL_matrix[self.d2i[d], self.w2i[w]] for w in known_words]
        return sum(relevances) / len(known_words)

    def POP(self, d, query):
        """
        Get aggregated popularity of document d for all words in query.
        Popularity measures how dominant this document is among all documents for these query words.
        
        Args:
            d: document infohash
            query: search query string
        Returns:
            float: average popularity score across all known words in query
        """
        if d not in self.d2i:
            return 0.0
        
        words = query.lower().split()
        known_words = [w for w in words if w in self.w2i]
        if not known_words:
            return 0.0
        
        popularities = [self.POP_matrix[self.d2i[d], self.w2i[w]] for w in known_words]
        return sum(popularities) / len(known_words)

    def matching_score(self, b, d):
        """Get matching score between user b and document d"""
        if b not in self.b2i or d not in self.d2i:
            return 0.0
        return self.matching_scores[self.b2i[b], self.d2i[d]]
