import argparse
import os, sys, time
import json
import jsonlines
import pandas as pd
import pickle
from utils import timed_func


def main(args):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    category_counts_df = pd.read_csv("count_categorize.csv")
    category_counts_df.drop(columns=["Unnamed: 0"], inplace=True)
    categories = list(category_counts_df)

    for i in range(100):
        start_time = time.time()
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        for cat in categories:
            cat_dir = os.path.join(args.output_dir, cat)
            if not os.path.exists(cat_dir):
                os.makedirs(cat_dir)

        # metadata
        md_fname = os.path.join(args.input_dir, "metadata", f"metadata_{i}.jsonl")
        grouped_metadata = {}
        for cat in categories:
            grouped_metadata[cat] = []
        with open(md_fname, "r") as f_meta:
            for line in f_meta:
                meta_dict = json.loads(line)

                if not meta_dict["has_pdf_parse"]:
                    continue
                if meta_dict["mag_field_of_study"] is None:
                    continue 
                cat = meta_dict["mag_field_of_study"][0]
                grouped_metadata[cat].append(meta_dict)

        for cat in grouped_metadata:
            with open(os.path.join(args.output_dir, cat, f"metadata_{i}.jsonl"), "w") as fo_meta:
                for item in grouped_metadata[cat]:
                    fo_meta.write(json.dumps(item))
                    fo_meta.write("\n")

        # pdf_parses
        pp_fname = os.path.join(args.input_dir, "pdf_parses", f"pdf_parses_{i}.jsonl")
        indexed_pdf = {}
        with open(pp_fname, "r") as f_pp:
            for line in f_pp:
                pp_dict = json.loads(line)
                indexed_pdf[pp_dict['paper_id']] = pp_dict 
        for cat in grouped_metadata:
            cat_pdf = {}
            for item in grouped_metadata[cat]:
                if item['paper_id'] in indexed_pdf:
                    cat_pdf[item['paper_id']] = indexed_pdf[item['paper_id']]
            with open(os.path.join(args.output_dir, cat, f"pdf_parses_{i}.pkl"), "wb") as fo_pp:
                pickle.dump(cat_pdf, fo_pp)

        # Bookkeep
        print ("Chunk {} done in {:.2f} seconds".format(i, 
            time.time() - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../../data/S2ORC/20200705v1/full/")
    parser.add_argument("--output_dir", type=str, default="../../data/S2ORC/20200705v1/by_category")
    args = parser.parse_args()
    print(args)
    main(args)