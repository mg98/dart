import pickle
from sklearn.metrics import ndcg_score
import numpy as np
from baselines.panache import panache_rank
from baselines.ltr import ltr_rank

def calculate_ndcg(ua):
    true_relevance = [1 if i == ua.chosen_index else 0 for i in range(len(ua.results))]
    predicted_relevance = [1/np.log2(i+2) for i in range(len(ua.results))]  # Assuming rank-based relevance
    return ndcg_score([true_relevance], [predicted_relevance])

if __name__ == "__main__":
    with open('user_activities.pkl', 'rb') as f:
        user_activities = pickle.load(f)

    average_ndcg = np.mean([calculate_ndcg(ua) for ua in user_activities])
    print(f"Average nDCG: {average_ndcg}")

    panache_rank(user_activities)

    average_ndcg = np.mean([calculate_ndcg(ua) for ua in user_activities])
    print(f"Average nDCG: {average_ndcg}")

    ltr_rank(user_activities)

    average_ndcg = np.mean([calculate_ndcg(ua) for ua in user_activities])
    print(f"Average nDCG: {average_ndcg}")