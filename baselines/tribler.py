from common import ranking_func

@ranking_func(shuffle=False)
def tribler_rank(_, activities):
    return activities
