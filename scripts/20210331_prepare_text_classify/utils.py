import time
import pandas as pd
from pathlib import Path

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

def get_venue_names(option='EMNLP'):
    """
    Input: option: str
    Output: a dictionary, with keys the venue names on this option.
    The values are 
    """
    base_path = "../../data/venue_name_labels"
    df = pd.read_csv(Path(base_path, f"{option}.csv"))
    D = {}
    for i, row in df.iterrows():
        D[row.venue] = row.label
    return D