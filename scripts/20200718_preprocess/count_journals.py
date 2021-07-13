import json
import os, sys, time
import pandas as pd 


def count_venue_journals():
    print ("Counting journals... ")
    data_dir = "../../data/S2ORC/20200705v1/by_category"
    all_journals = {}
    for cat in os.listdir(data_dir):
        start_time = time.time()
        cat_dir = os.path.join(data_dir, cat)
        cat_journals = {}
        for i in range(100):
            md_fname = os.path.join(data_dir, cat, f"metadata_{i}.jsonl")
            with open(md_fname, "r") as f_md:
                for line in f_md:
                    md_dict = json.loads(line)
                    venue = "None"
                    if md_dict["journal"] is not None:  # Prioritize journals
                        venue = md_dict["journal"]
                    elif md_dict["venue"] is not None:
                        venue = md_dict["venue"]
                    
                    if venue not in cat_journals:
                        cat_journals[venue] = 1
                    else:
                        cat_journals[venue] += 1
        all_journals[cat] = cat_journals 

        print ("{} has {} journals (counted in {:.2f} seconds)".format(
            cat, 
            len(cat_journals),
            time.time() - start_time))

    with open("journals_count.json", "w") as f:
        f.write(json.dumps(all_journals))

count_venue_journals()