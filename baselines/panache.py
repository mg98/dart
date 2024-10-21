from collections import defaultdict
from common import UserActivity

# Do I simulate the ranking loss based on past or all user activities?

def panache_rank(user_activities: list[UserActivity]):
    """
    Implementing Panaché's ranking by keyword hit counts.
    The authors furthermore suggest to randomize the order if hit counts are low (not implemented here).
    """
    # count how many times each infohash is hit for each keyword
    hit_counts = defaultdict(lambda: defaultdict(int)) # keyword -> infohash -> count
    for ua in user_activities:
        for keyword in ua.query.lower().split():
            hit_counts[keyword][ua.chosen_result.infohash] += 1

    # re-rank based on hit counts
    for ua in user_activities:
        ua.results.sort(key=lambda x: sum(hit_counts[k][x.infohash] for k in ua.query.lower().split()), reverse=True)

    return user_activities