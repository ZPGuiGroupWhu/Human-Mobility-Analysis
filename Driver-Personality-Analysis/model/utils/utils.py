def time2seconds(t):
    h = t.hour
    m = t.minute
    s = t.second
    return int(h) * 3600 + int(m) * 60 + int(s)


def seconds2time(t):
    h = t // 3600
    m = (t - (3600 * h)) // 60
    s = t - (3600 * h) - (60 * m)
    return h, m, s
