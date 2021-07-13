import nltk
import re 
import os 
from pathlib import Path
import pandas as pd 
import argparse

from utils import timed_func 

rst_signals = [
    "Attribution", "Background", "Cause", "Comparison", "Condition", "Contrast", "Elaboration", "Enablement", "Evaluation", "Explanation", "Joint", "Manner-Means", "Topic-Comment", "Summary", "Temporal", "Topic-Change", "textual-organization", "same-unit"
]

def is_legal_signal(label):
    # nltk.tree treats small brackets (in texts) as nodes. Filter out any nodes that do not end with [w][w], where w can be any word character.
    return re.search(r"\[[A-Z]\]\[[A-Z]\]$", label) is not None

def count_rst_features(parsed_fname):
    """
    Adopted from RST-probe plotting notebook.
    """
    with open(parsed_fname, "r") as f:
        s = f.read()
    try:
        root = nltk.tree.Tree.fromstring(s)
    except ValueError as e:
        return None 

    # The child of the tree is either str (word) or node (signals).
    feats = {}
    total_sig_count = 0
    for sig in rst_signals:
        feats[sig] = 0
    positions = root.treepositions()
    for i, pos in enumerate(positions):
        node = root[pos]
        
        if not isinstance(node, str):
            sig_ = node.label()  # e.g., "Elaboration[N][S]"
            if not is_legal_signal(sig_):
                continue 
            sig = sig_[:-6]
            
            if sig in rst_signals:
                feats[sig] += 1
                total_sig_count += 1
            else:
                print (f"Signal {sig} not in rst_signals. Skipping...")
    
    total_sig_count = max(1, total_sig_count)  # Avoid divide-by-zero error
    for sig in feats:
        feats[sig] /= total_sig_count 
    return feats 


@timed_func
def parse_rst_features(args):
    chunkid = args.chunk_id 

    ret = {'paper_id': []}
    for sig in rst_signals:
        ret[f"rst_{sig}"] = []

    parse_result_dir = f"feng-hirst-rst-parser/texts/results/chunk_{chunkid}"
    fnames = [fn for fn in Path(parse_result_dir).glob("*.tree")]
    print ("{} contains {} parsed trees".format(parse_result_dir, len(fnames)))
    skipped = 0
    for fname in fnames:
        feats = count_rst_features(fname)
        if feats is None:
            #print ("Unable to read parsed results: {}".format(fname))
            skipped += 1
            continue 
        ret['paper_id'].append(fname.stem.split(".")[0])
        for sig in feats:
            ret[f"rst_{sig}"].append(feats[sig])

    df = pd.DataFrame(ret)
    df.to_csv(args.export_path, index=False)
    print ("Extracted RST features from {} docs. Skipped {} docs.".format(len(df), skipped))


parser = argparse.ArgumentParser()
parser.add_argument('--chunk_id', type=int, default=0)
parser.add_argument('--export_path', type=str, default='./test_rst_feat.csv')
args = parser.parse_args()
parse_rst_features(args)
