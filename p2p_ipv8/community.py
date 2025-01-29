from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from ipv8.community import Community, CommunitySettings
from ipv8.lazy_community import lazy_wrapper
from ipv8.messaging.payload_dataclass import DataClassPayload

from common import UserActivity, UserActivityTorrent, TorrentInfo

if TYPE_CHECKING:
    from ipv8.types import Peer


@dataclass
class USATPayload(DataClassPayload):
    """
    Users' Serialized Activity Torrent Payload.
    """
    infohash: str
    seeders: int
    leechers: int
    has_torrent_info: bool
    title: str
    tags: list[str]
    timestamp: float
    size: int

    @classmethod
    def from_ua_torrent(cls, ua: UserActivityTorrent) -> type[Self]:
        return cls(ua.infohash, ua.seeders, ua.leechers,
                   ua.torrent_info is not None,
                   "" if (ua.torrent_info is None) else ua.torrent_info.title,
                   [] if (ua.torrent_info is None) else ua.torrent_info.tags,
                   0.0 if (ua.torrent_info is None) else ua.torrent_info.timestamp,
                   0 if (ua.torrent_info is None) else ua.torrent_info.size)

    def to_user_activity(self) -> UserActivityTorrent:
        out = UserActivityTorrent({"infohash": self.infohash, "seeders": self.seeders, "leechers": self.leechers})
        if self.has_torrent_info:
            out.torrent_info = TorrentInfo(title=self.title, tags=self.tags, timestamp=self.timestamp, size=self.size)
        return out

@dataclass
class USAPayload(DataClassPayload[1]):
    """
    Users' Serialized Activity Payload.
    """

    issuer: str
    query: str
    timestamp: int
    results: list[USATPayload]
    chosen_index: int

    @classmethod
    def from_user_activity(cls, ua: UserActivity) -> type[Self]:
        return cls(ua.issuer, ua.query, ua.timestamp,
                   [USATPayload.from_ua_torrent(result) for result in ua.results],
                   ua.chosen_index)

    def to_user_activity(self) -> UserActivity:
        return UserActivity({
            "issuer": self.issuer,
            "query": self.query,
            "timestamp": self.timestamp * 1000,
            "results": [result.to_user_activity() for result in self.results],
            "chosen_index": self.chosen_index
        })


class USASettings(CommunitySettings):
    """
    Add any settings here if you need them.
    """


class USACommunity(Community):

    community_id = b"\x00\x00USAGossipCommunity"
    settings_class = USASettings

    def __init__(self, settings: USASettings) -> None:
        super().__init__(settings)

        self.add_message_handler(USAPayload, self.on_usa)

    def received_user_activity(self, ua: UserActivity, from_peer: Peer) -> None:
        """
        TODO: do something here.
        """

    def send_user_activity(self, ua: UserActivity, to: Peer) -> None:
        self.ez_send(to, USAPayload.from_user_activity(ua))

    @lazy_wrapper(USAPayload)
    def on_usa(self, peer: Peer, payload: USAPayload) -> None:
        self.received_user_activity(payload.to_user_activity(), peer)
