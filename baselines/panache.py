from collections import defaultdict
from common import UserActivity, ranking_func
from functools import cache

def compute_hit_counts(activities: list[UserActivity]):
    """
    Compute the hit counts for each keyword in each query.
    """
    hit_counts = defaultdict(lambda: defaultdict(int)) # keyword -> infohash -> count
    for ua in activities:
        for keyword in ua.query.lower().split():
            hit_counts[keyword][ua.chosen_result.infohash] += 1
    return hit_counts

@ranking_func
def panache_rank(clicklogs: list[UserActivity], activities: list[UserActivity] = None):
    """
    Optimized Panaché's ranking by keyword hit counts.
    """
    hit_counts = compute_hit_counts(clicklogs)

    @cache
    def cached_query_split(query):
        return query.lower().split()

    # Re-rank activities
    for ua in activities:
        keywords = cached_query_split(ua.query)
        ua.results.sort(
            key=lambda x: sum(hit_counts[k][x.infohash] for k in keywords),
            reverse=True
        )

    return activities
