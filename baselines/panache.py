from collections import defaultdict
from common import UserActivity, ranking_func
from functools import cache

@ranking_func
def panache_rank(clicklogs: list[UserActivity], activities: list[UserActivity] = None):
    """
    Implementing Panaché's ranking by keyword hit counts.
    The authors furthermore suggest to randomize the order if hit counts are low (not implemented here).
    """
    # count how many times each infohash is hit for each keyword
    hit_counts = defaultdict(lambda: defaultdict(int)) # keyword -> infohash -> count
    for ua in clicklogs:
        for keyword in ua.query.lower().split():
            hit_counts[keyword][ua.chosen_result.infohash] += 1

    # re-rank based on hit counts
    for ua in activities:
        ua.results.sort(key=lambda x: sum(hit_counts[k][x.infohash] for k in ua.query.lower().split()), reverse=True)

    return activities


@ranking_func
def panache_rank_fast(clicklogs: list[UserActivity], activities: list[UserActivity] = None):
    """
    Optimized Panaché's ranking by keyword hit counts.
    """

    # Precompute keyword -> infohash -> count
    hit_counts = defaultdict(lambda: defaultdict(int))
    for ua in clicklogs:
        keywords = ua.query.lower().split()
        infohash = ua.chosen_result.infohash
        for keyword in keywords:
            hit_counts[keyword][infohash] += 1

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
