from collections import Counter
from common import UserActivity

def grank(user_activities: list[UserActivity]):
    """
    Implementing GRank's similarity-based ranking.
    """
    users = set(ua.issuer for ua in user_activities)

    def similarity_score(user_a: str, user_b: str):
        """
        Calculate the similarity score between two users $S_i(n_j)$.
        """

        def get_top_k_terms(user, k=5):
            all_terms = [term for ua in user_activities if ua.issuer == user for term in ua.query.lower().split()]
            return set(term for term, _ in Counter(all_terms).most_common(k))

        # Find top K most frequent query terms for user_a and user_b
        user_a_terms = get_top_k_terms(user_a)
        user_b_terms = get_top_k_terms(user_b)
        
        # Calculate the intersection of terms
        common_terms = user_a_terms.intersection(user_b_terms)

        # Find clicked results for user_a and user_b
        user_a_clicks = set(ua.chosen_result.infohash for ua in user_activities if ua.issuer == user_a and ua.chosen_result)
        user_b_clicks = set(ua.chosen_result.infohash for ua in user_activities if ua.issuer == user_b and ua.chosen_result)
        
        common_clicks = user_a_clicks.intersection(user_b_clicks)
        union_clicks = user_a_clicks.union(user_b_clicks)
        
        if len(common_clicks) == 0:
            return 0
        
        return (len(common_terms) + len(union_clicks) ** 2) / len(common_clicks)

    user_similarities = {user_a: {
        user_b: similarity_score(user_a, user_b) for user_b in users if user_a != user_b
    } for user_a in users}

    def rank_results(result):
        F = 0
        return sum((result.seeders * user_sim + F) for user_sim in user_similarities[ua.issuer])
    
    for ua in user_activities:
        ua.results.sort(key=rank_results, reverse=True)
    
    return user_activities