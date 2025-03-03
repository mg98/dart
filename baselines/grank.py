from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from utils.common import UserActivity, ranking_func
from functools import cache

@ranking_func
def grank(clicklogs: list[UserActivity], activities: list[UserActivity] = None):
    """
    Implementing GRank's similarity-based ranking.
    """
    users = set(ua.issuer for ua in clicklogs)

    term_counts = {
        user: Counter(
            term for ua in clicklogs if ua.issuer == user for term in ua.query.lower().split()
        )
        for user in users
    }

    @cache
    def get_top_k_terms(user, k=5):
        return set(term for term, _ in term_counts[user].most_common(k))

    @cache
    def similarity_score(user_a: str, user_b: str):
        """
        Calculate the similarity score between two users $S_i(n_j)$.
        """

        # performance tweak as parameters are symmetric
        if user_a > user_b:
            return similarity_score(user_b, user_a)
        
        # Find top K most frequent query terms for user_a and user_b
        user_a_terms = get_top_k_terms(user_a)
        user_b_terms = get_top_k_terms(user_b)
        
        # Calculate the intersection of terms
        common_terms = user_a_terms.intersection(user_b_terms)

        # Find clicked results for user_a and user_b
        user_a_clicks = set(ua.chosen_result.infohash for ua in clicklogs if ua.issuer == user_a and ua.chosen_result)
        user_b_clicks = set(ua.chosen_result.infohash for ua in clicklogs if ua.issuer == user_b and ua.chosen_result)
        
        common_clicks = user_a_clicks.intersection(user_b_clicks)
        union_clicks = user_a_clicks.union(user_b_clicks)
        
        if len(common_clicks) == 0:
            return 0
        
        return (len(common_terms) + len(union_clicks) ** 2) / len(common_clicks)

    user_similarities = {user_a: {
        user_b: similarity_score(user_a, user_b) for user_b in users if user_a != user_b
    } for user_a in users}

    # Precompute click counts for each infohash
    click_counts = {}
    for ua in clicklogs:
        if ua.chosen_result:
            click_counts[ua.chosen_result.infohash] = click_counts.get(ua.chosen_result.infohash, 0) + 1

    @cache
    def rank_results(infohash, issuer, F=0):
        sims = {user: sim for user, sim in user_similarities[issuer].items() if sim != 0}
        return sum(click_counts.get(infohash, 0) * (sim + F) for user, sim in sims.items())
    
    for ua in activities:
        ua.results.sort(key=lambda x: rank_results(x.infohash, ua.issuer), reverse=True)
    
    return activities

def precompute_grank_score_fn(clicklogs: list[UserActivity]) -> callable:
    users = set(ua.issuer for ua in clicklogs)
    
    # Preprocess user data
    term_counts = defaultdict(Counter)
    user_clicks = defaultdict(set)
    click_counts = Counter()

    for ua in clicklogs:
        term_counts[ua.issuer].update(ua.query.lower().split())
        if ua.chosen_result:
            user_clicks[ua.issuer].add(ua.chosen_result.infohash)
            click_counts[ua.chosen_result.infohash] += 1
    
    # Precompute similarities
    def compute_similarity(user_a):
        sims = {}
        user_a_terms = set(term_counts[user_a].keys())
        user_a_clicks = user_clicks[user_a]
        for user_b in users - {user_a}:
            user_b_terms = set(term_counts[user_b].keys())
            user_b_clicks = user_clicks[user_b]
            
            common_terms = user_a_terms & user_b_terms
            common_clicks = user_a_clicks & user_b_clicks
            union_clicks = user_a_clicks | user_b_clicks

            if common_clicks:
                sims[user_b] = (len(common_terms) + len(union_clicks) ** 2) / len(common_clicks)
        return user_a, sims

    with ThreadPoolExecutor() as executor:
        user_similarities = dict(executor.map(compute_similarity, users))

    def grank_score(issuer, infohash, F=0):
        sims = user_similarities.get(issuer, {})
        return sum(click_counts[infohash] * (sim + F) for sim in sims.values() if sim != 0)
    
    return grank_score

@ranking_func
def grank_fast(clicklogs: list[UserActivity], activities: list[UserActivity] = None):
    grank_score = precompute_grank_score_fn(clicklogs)
    for ua in activities:
        ua.results.sort(key=lambda x: grank_score(ua.issuer, x.infohash), reverse=True)
    
    return activities
