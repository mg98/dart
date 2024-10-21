class TorrentInfo:
    title: str
    tags: list[str]
    creation_date: str
    size: int

    def __str__(self):
        return f"Title: {self.title}, Tags: {self.tags}, Creation Date: {self.creation_date}, Size: {self.size}"

class UserActivityTorrent:
    infohash: str
    seeders: int
    leechers: int
    torrent_info: TorrentInfo

    def __str__(self):
        return f"Infohash: {self.infohash}, Seeders: {self.seeders}, Leechers: {self.leechers}, Torrent Info: {self.torrent_info}"

class UserActivity:
    issuer: str
    query: str
    chosen_index: int
    timestamp: int
    results: list[UserActivityTorrent]

    @property
    def chosen_result(self) -> UserActivityTorrent:
        return self.results[self.chosen_index]

    def __str__(self):
        return f"Issuer: {self.issuer}, Query: {self.query}, Chosen Index: {self.chosen_index}, Timestamp: {self.timestamp}, Results: {self.results}"
