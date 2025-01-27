import json
import os
import pickle
import time
from common import *

metadata_db_path = os.path.expanduser('./metadata.db')

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
            if data["chosen_index"] == -1:
                continue
            data['issuer'] = user
            ua = UserActivity(data)
            user_activities.append(ua)

fetch_torrent_infos(user_activities)

# Filter invalid user activities
user_activities = [ua for ua in user_activities 
                   if ua.chosen_result.torrent_info is not None 
                   and ua.query.strip() != "" 
                   and ua.timestamp != 0 
                   and ua.timestamp < time.time()]

# Filter out results whose torrent info is missing
for ua in user_activities:
    ua.query = ua.query.strip().lower()
    ua.results = [res for res in ua.results if res.torrent_info is not None]

user_activities = list(filter(lambda ua: len(ua.results) > 1, user_activities))

for ua in user_activities:
    for doc in ua.results:
        assert doc.torrent_info is not None

# Persist user_activities to disk
with open('user_activities.pkl', 'wb') as f:
    pickle.dump(user_activities, f)

print(f"User activities saved to user_activities.pkl")
