import argparse
import json 
import os, sys, time 
import pandas as pd 
from utils import timed_func


def extract_venue_info(mds, args):
    res = {'paper_id': [], 'venue': [], 'venue_category': [], 'venue_is_top': []}

    venue_df = pd.read_csv(args.venue_label)
    venue_dict = {}
    for i, row in venue_df.iterrows():
        venue_dict[row['venue']] = (_row_to_category(row), row['label'])

    for j, md in enumerate(mds):
        journal = md.get('journal', None)
        venue = md.get('venue', None)
        if journal is not None:
            v = journal 
        elif venue is not None:
            v = venue 
        else:
            v = "None"

        category, top = venue_dict.get(v, ("Other", 0))
        res['paper_id'].append(md['paper_id'])
        res['venue'].append(v)
        res['venue_is_top'].append(top)
        res['venue_category'].append(category)

    return pd.DataFrame(res) 


def _row_to_category(vdf_row):
    cats = ['NLP', 'Speech', 'ML', 'AI', 'CV', 'Robo']
    res = []
    for c in cats:
        if vdf_row[c]:
            res= c
    # Just go with the last matched category. i.e., if a venue name is in both 'AI' and 'Robo' categories, take 'Robo'.
    return res


parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default="../../data/S2ORC/20200705v1/by_category/Computer Science/")
parser.add_argument("--venue_label", type=str, default="../../data/df_ai_labeled.csv")
parser.add_argument("--export", type=str, default="venue_info.csv")
args = parser.parse_args()
print(args)

all_venue_info = None 
start_time = time.time()
for chunk in range(100):
    with open(os.path.join(args.base_dir, "metadata_{}.jsonl".format(chunk)), "r") as f_md:
        mds = [json.loads(line) for line in f_md.readlines()]

    venue_info_df = extract_venue_info(mds, args)
    if all_venue_info is None:
        all_venue_info = venue_info_df 
    else:
        all_venue_info = pd.concat([all_venue_info, venue_info_df])
    print ("Chunk {} done. Time elapsed: {:.2f} seconds".format(chunk, time.time() - start_time))
all_venue_info.to_csv(args.export, index=False)
