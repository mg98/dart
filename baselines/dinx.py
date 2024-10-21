from common import UserActivity

def dinx_rank(user_activities: list[UserActivity]):
    """
    Implementing DINX's ranking by popularity score measured by the number of seeders.
    """
    for ua in user_activities:
        ua.results.sort(key=lambda x: x.seeders, reverse=True)
        
    return user_activities