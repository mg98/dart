from collections import Counter
from common import UserActivity, ranking_func

def compute_click_counts(clicklogs: list[UserActivity]):
    return Counter(ua.chosen_result.infohash for ua in clicklogs if ua.chosen_result)

@ranking_func
def dinx_rank(clicklogs: list[UserActivity], activities: list[UserActivity] = None):
    """
    Implementing DINX's ranking by popularity score measured by click counts.
    """
    click_counts = compute_click_counts(clicklogs)
    for ua in activities:
        ua.results.sort(key=lambda x: click_counts[x.infohash], reverse=True)
        
    return activities

@ranking_func
def dinx_rank_by_seeders(_: list[UserActivity], activities: list[UserActivity] = None):
    """
    Implementing DINX's ranking by popularity score measured by the number of seeders.
    Uses infohash as a deterministic tiebreaker.
    """
    for ua in activities:
        ua.results.sort(key=lambda x: (x.seeders, x.infohash), reverse=True)
        
    return activities