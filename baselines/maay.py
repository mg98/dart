from collections import defaultdict
from common import UserActivity
import math

def hoeffding_bound(n, delta=0.05, a=0, b=1):
    """
    Compute the Hoeffding bound epsilon(n) for a given number of samples n.
    
    Parameters:
    n (int): Number of observations.
    delta (float): Confidence level (e.g., 0.05 for 95% confidence).
    a (float): Lower bound of the random variable.
    b (float): Upper bound of the random variable.
    
    Returns:
    float: The Hoeffding bound epsilon(n).
    """
    range_squared = (b - a) ** 2
    epsilon = math.sqrt((range_squared * math.log(2 / delta)) / (2 * n))
    return epsilon

def maay_rank(user_activities: list[UserActivity]):
    """
    Implementing MAAY's ranking by keyword hit counts.
    """

    W = set(w for ua in user_activities for w in ua.query.lower().split())
    D = set(ua.chosen_result.infohash for ua in user_activities)

    # def W(a):
    #     """
    #     Set of words known by node `a`.
    #     This set contains the words occurring in the documents known by `a` 
    #     and the words for which `a` has a knowledge through previous incoming messages.
    #     """
    #     return set(w for ua in user_activities if ua.issuer == a for w in ua.query.lower().split())

    # def D(a):
    #     """
    #     Set of documents stored by node `a`.
    #     """
    #     return set(ua.chosen_result.infohash for ua in user_activities if ua.issuer == a)

    def claim(b, w):
        """
        The number of claims for word `w` expressed by node `b` and noticed by node `a`.

        A node `a` has four ways to notice that a neighbor `b` claims its interest for a word `w`:
        1. `b` sends a search request on `w`,
        2. `b` sends back a search response on `w`,
        3. `b` is a document provider for a search on `w`,
        4. node `a` provides to `b` a document on `w`.
        """
        return sum(1 for ua in user_activities if ua.issuer == b and w in ua.query.lower().split())

    def vote(d, w):
        """
        The number of download requests received by node `a` for document `d` and word `w`.
        """
        return sum(1 for ua in user_activities if ua.chosen_result.infohash == d and w in ua.query.lower().split())

    def SP(b, w):
        return claim(b, w) / sum(claim(b, w2) for w2 in W)

    def REL(d, w):
        return vote(d, w) / sum(vote(d, w2) for w2 in W) - hoeffding_bound(sum(vote(d, w2) for w2 in W))

    def POP(d, w):
        return vote(d, w) / sum(vote(d2, w) for d2 in D) - hoeffding_bound(sum(vote(d2, w) for d2 in D))

    def matching(z, d):
        return sum(REL(d, w) * SP(z, w) for w in W)

    def DRS(z, d, w):
        return REL(d, w) * POP(d, w) * matching(z, d)
    
    # re-rank based on DRS score
    for ua in user_activities:
        ua.results.sort(key=lambda x: DRS(ua.issuer, x.infohash, x.infohash, x.infohash), reverse=True)

    return user_activities