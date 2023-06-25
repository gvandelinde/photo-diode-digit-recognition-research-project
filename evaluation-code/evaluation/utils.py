import time

def get_timestr():
    # Time string from: https://stackoverflow.com/a/10607768
    timestr = time.strftime("%Y%m%d-%H%M%S")
    return timestr