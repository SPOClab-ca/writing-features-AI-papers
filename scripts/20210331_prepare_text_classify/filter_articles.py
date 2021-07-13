import argparse 
import os, sys, time
import json
import jsonlines 
import nltk
from nltk import word_tokenize, sent_tokenize
import pandas as pd 
from pathlib import Path
import pickle 
from utils import timed_func 


categories = [
        "AAAI", "ACL", "COLING", "CVPR", "EMNLP", "ICML", "ICRA", "IJCAI", "NAACL", "NIPS", "ICASSP"
    ]

def get_venue_labels(args):
    """
    Return venue_name -> [v_shortname, label]
    where venue_name is e.g., "AAAI Spring Symposium 2013"
    v_shortname is e.g., "AAAI"
    label is binary (0 - Workshop, 1 - Conference)
    """
    venue_name_labels_map = {} 
    
    for v_shortname in categories:
        if args.include_arxiv:
            fname = Path(args.venue_name_labels_path, f"{v_shortname}_v_arxiv.csv")
        else:
            fname = Path(args.venue_name_labels_path, f"{v_shortname}.csv")
    
        df = pd.read_csv(fname) 
        for i, row in df.iterrows():
            v = row.venue 
            venue_name_labels_map[v] = [v_shortname, row.label]
    return venue_name_labels_map 


def prepare_text(metadata, pdfparse):
    """
    Return:
        abstract: str (or None)
        bodytext: list of str (or None)
    """
    MIN_WORDS = 5
    abstract = metadata['abstract']
    if abstract is None or len(abstract.split()) < MIN_WORDS:
        return None, None
    bodytext = []
    for section in pdfparse['body_text']:
        if section['text'] is not None:
            bodytext.extend(sent_tokenize(section['text']))
    bodytext = [sent for sent in filter(lambda s: len(s)>0, bodytext)]
    if len(bodytext)  == 0:
        return None, None
    return abstract, bodytext


@timed_func 
def filter_articles_main(venue_name_labels_map, args):
    """
    Find these articles. Save into standalone pkl file

    results: dictionary (key is category)
    results['AAAI']: list of articles. Each article contains {'abstract': str, 'bodytext': list of str, 'venue': str, 'label': int (0 or 1)}  
    """
    
    results = {}    
    for cat in categories:
        results[cat] = []

    # Traverse the CompSci paper collections. Save 
    for chunk_id in range(100):
        start_time = time.time()
        with open(Path(args.input_dir, f"metadata_{chunk_id}.jsonl"), "r") as f_md:
            mds = [json.loads(line) for line in f_md.readlines()]

        with open(Path(args.input_dir, f"pdf_parses_{chunk_id}.pkl"), "rb") as f_pp:
            cat_pdf = pickle.load(f_pp)

        skipped = 0
        collected = 0
        for i, metadata in enumerate(mds):
            paper_id = metadata['paper_id']
            pdfparse = cat_pdf[paper_id]
            abstract, bodytext = prepare_text(metadata, pdfparse)
            if abstract is None or metadata['year'] is None:
                skipped += 1 
                continue 

            # Following the 20201206_venue_info/venue_info.py convention (i.e., prioritize journal, then venue) for extracting venue information
            journal = metadata.get('journal', None)
            venue = metadata.get('venue', None) 
            if journal is not None:
                v = journal 
            elif venue is not None:
                v = venue 
            else:
                v = "None"
            
            if v in venue_name_labels_map:
                shortname, label = venue_name_labels_map[v]

                curr_year = 2021
                citation_per_year = len(metadata['inbound_citations']) / (curr_year - metadata['year'])
                results[shortname].append({
                    "abstract": abstract,
                    "bodytext": bodytext,
                    "venue": v,
                    "label": label,
                    "year": metadata.get('year', None),
                    "title": metadata.get("title", None),
                    "annual_citations": citation_per_year
                })
                collected += 1
        print ("Chunk {} done in {:.2f} seconds. Skipped {} entries. Collected {} entries.".format(chunk_id, time.time() - start_time, skipped, collected))

    export_dir = Path(args.export)
    with open(args.export, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue_name_labels_path", type=str, default="../../data/venue_name_labels")
    parser.add_argument("--include_arxiv", action="store_true", default=False)
    parser.add_argument("--input_dir", type=str, default="../../data/S2ORC/20200705v1/by_category/Computer Science/")
    parser.add_argument("--export", type=str, default="../../data/text_classify_articles_with_arxiv.pkl") 

    args = parser.parse_args()
    print(args) 

    name_map = get_venue_labels(args)
    filter_articles_main(name_map, args)
