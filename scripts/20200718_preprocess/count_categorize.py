import argparse
import os, sys, time
import pandas as pd
import json
import jsonlines
from utils import timed_func


@timed_func
def read_metadata(fname):
    doc_cat_counts = {}  # dict. str -> int
    with open(fname) as f_meta:
        for line in f_meta:
            meta_dict = json.loads(line)
            if not meta_dict['has_pdf_parse']:
                continue
            if meta_dict['mag_field_of_study'] is None:
                continue
            for cat in meta_dict['mag_field_of_study']:
                if cat not in doc_cat_counts:
                    doc_cat_counts[cat] = 1
                else:
                    doc_cat_counts[cat] += 1
    return doc_cat_counts 


def main(args):
    metadata_dir = os.path.join(args.input_dir, "metadata")
    pdf_parses_dir = os.path.join(args.input_dir, "pdf_parses")

    counts_by_chunk = []  # list (len n. chunk) of dict (str -> int)
    for fn in os.listdir(metadata_dir):
        fname = os.path.join(metadata_dir, fn)
        doc_cat_counts = read_metadata(fname)
        counts_by_chunk.append(doc_cat_counts)

    dfdata = {}  # dict (str -> list)
    for doc_cat_counts in counts_by_chunk:
        for cat in doc_cat_counts.keys():
            if cat not in dfdata:
                dfdata[cat] = []
    for doc_cat_counts in counts_by_chunk:
        for cat in dfdata.keys():
            if cat in doc_cat_counts:
                dfdata[cat].append(doc_cat_counts[cat])
            else:
                dfdata[cat].append(0)
    df = pd.DataFrame(dfdata)
    df.to_csv("count_categorize.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../../data/S2ORC/20200705v1/full/")
    parser.add_argument("--output_dir", type=str, default="test_output")
    args = parser.parse_args()
    print(args)
    main(args)