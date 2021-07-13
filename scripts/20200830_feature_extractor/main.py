import argparse 
import json 
import os
import pandas as pd 
import pickle
from extract_lex_features import run_lex_features_extraction
#from extract_grammar_features import run_grammar_features_extraction
#from extract_rst_features import run_rst_features_extraction
from extract_surprisal_features import run_surprisal_features_extraction
from extract_syntax_features import run_syntax_features_extraction


parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default="../../data/S2ORC/20200705v1/by_category/Computer Science/")
parser.add_argument("--chunk", type=int, default=0)
parser.add_argument("--export", type=str, default="feature_v2/chunk0.csv")
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--checkpoint", type=str, default="checkpoint.pkl")
parser.add_argument("--slurm_jid", type=int, default=100000)
parser.add_argument("--include_lex", action="store_true")
parser.add_argument("--include_grammar", action="store_true")
parser.add_argument("--include_rst", action="store_true")
parser.add_argument("--include_surprisal", action="store_true")
parser.add_argument("--include_syntax", action="store_true")
args = parser.parse_args()
print(args)

with open(os.path.join(args.base_dir, "pdf_parses_{}.pkl".format(args.chunk)), "rb") as f_pp:
    cat_pdf = pickle.load(f_pp)
    
with open(os.path.join(args.base_dir, "metadata_{}.jsonl".format(args.chunk)), "r") as f_md:
    mds = [json.loads(line) for line in f_md.readlines()]

if os.path.exists(args.checkpoint):
    with open(args.checkpoint, "rb") as f:
        checkpoint = pickle.load(f)
else:
    checkpoint = None 

def _join_df(prev, df):
    if prev is None:
        return df
    else:
        return pd.merge(prev, df, on="paper_id", how="outer") 

result_df = None 
if args.include_lex:
    lex_feat_df, result_df = run_lex_features_extraction(mds, cat_pdf, args, checkpoint=checkpoint, prev_result=result_df)
    result_df = _join_df(result_df, lex_feat_df)
if args.include_grammar:
    raise ValueError("GECTOR grammar model needs a previous transformer version. Use 20201109_gector scripts!")
    grammar_feat_df, result_df = run_grammar_features_extraction(mds, cat_pdf, args, checkpoint=checkpoint, prev_result=result_df)
    result_df = _join_df(result_df, grammar_feat_df)
if args.include_rst:
    raise ValueError("This RST extraction script is broken. Use 20201017_rst scripts!")
    rst_feat_df, result_df = run_rst_features_extraction(mds, cat_pdf, args, checkpoint=checkpoint, prev_result=result_df)
    result_df = _join_df(result_df, rst_feat_df)
if args.include_surprisal:
    surprisal_feat_df, result_df = run_surprisal_features_extraction(mds, cat_pdf, args, checkpoint=checkpoint, prev_result=result_df)
    result_df = _join_df(result_df, surprisal_feat_df)
if args.include_syntax:
    syntax_feat_df, result_df = run_syntax_features_extraction(mds, cat_pdf, args, checkpoint=checkpoint, prev_result=result_df)
    result_df = _join_df(result_df, syntax_feat_df)

result_df.to_csv(args.export, index=False)
