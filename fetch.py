import sqlite3
import json
import os
from common import UserActivity, UserActivityTorrent, TorrentInfo

# Get the path to the Tribler database
tribler_db_path = os.path.expanduser('~/.Tribler/8.0/sqlite/recommender.db')
metadata_db_path = os.path.expanduser('./metadata.db')

# Connect to the SQLite database
conn = sqlite3.connect(tribler_db_path)

cursor = conn.cursor()

# Query to select version and json fields from the Query table
query = "SELECT json FROM Query WHERE version = 0"

# Execute the query
cursor.execute(query)

# Fetch all results
results = cursor.fetchall()

# Process the results
user_activities = []
for row in results:

    # Parse the JSON data
    try:
        parsed_data = json.loads(row[0])
        
        # Validate and create UserActivity object
        try:
            user_activity = UserActivity()
            user_activity.query = parsed_data['query']
            user_activity.chosen_index = parsed_data['chosen_index']
            user_activity.results = []
            
            for result in parsed_data['results']:
                torrent = UserActivityTorrent()
                torrent.infohash = result['infohash']
                torrent.seeders = result['seeders']
                torrent.leechers = result['leechers']
                user_activity.results.append(torrent)
            
            user_activities.append(user_activity)
            
        except (KeyError, TypeError) as e:
            print(f"Error creating UserActivity object: {str(e)}")
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {str(e)}")

# Close the database connection
conn.close()

torrent_infos = dict()
for user_activity in user_activities:
    for result in user_activity.results:
        torrent_infos[result.infohash] = None

# Connect to the metadata database
conn = sqlite3.connect(metadata_db_path)
metadata_cursor = conn.cursor()

# Prepare the query to select all TorrentInfo fields at once
query = """
SELECT infohash, title, tags, timestamp/1000 as timestamp, size
FROM ChannelNode
WHERE infohash IN ({})
"""

# Convert infohashes to bytes and create placeholders for the SQL query
infohash_bytes = [bytes.fromhex(infohash) for infohash in torrent_infos]
placeholders = ','.join(['?' for _ in infohash_bytes])

# Execute the query with all infohashes at once
metadata_cursor.execute(query.format(placeholders), infohash_bytes)

# Fetch all results
results = metadata_cursor.fetchall()

# Process the results
for result in results:
    infohash = result[0].hex()  # Convert bytes back to hex string
    torrent_info = TorrentInfo()
    torrent_info.title = result[1]
    torrent_info.tags = result[2].split(',') if result[2] else []
    torrent_info.timestamp = result[3]
    torrent_info.size = result[4]
    torrent_infos[infohash] = torrent_info

# Handle any missing infohashes
for infohash in torrent_infos:
    if torrent_infos[infohash] is None:
        print(f"No result found for infohash: {infohash}")

# Assign torrent_info to each UserActivityTorrent
for user_activity in user_activities:
    for result in user_activity.results:
        result.torrent_info = torrent_infos[result.infohash]

conn.close()

import pickle

# Persist user_activities to disk
with open('user_activities.pkl', 'wb') as f:
    pickle.dump(user_activities, f)

print(f"User activities saved to user_activities.pkl")

# Example of how to load the data in another Python file:
# 
# import pickle
# 
# with open('user_activities.pkl', 'rb') as f:
#     loaded_user_activities = pickle.load(f)
# 
# print(f"Loaded {len(loaded_user_activities)} user activities")
