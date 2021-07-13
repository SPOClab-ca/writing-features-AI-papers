import time

def timed_func(foo):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        results = foo(*args, **kwargs)
        time_seconds = time.time() - start_time 
        print ("{} done in {:.2f} seconds ({:.2f} hours)".format(
            foo.__name__, time_seconds, time_seconds / 3600
        ))
        return results 
    return wrapper 