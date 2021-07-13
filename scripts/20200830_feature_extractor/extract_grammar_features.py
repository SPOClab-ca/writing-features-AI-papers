import pandas as pd
import pickle 
import numpy as np 
import os, sys, time
from pathlib import Path
import shlex, subprocess 
import shutil
import tempfile
from nltk.tokenize import sent_tokenize

from utils import timed_func


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
def run_grammar_features_extraction(mds, cat_pdf, args, checkpoint=None):
    if (checkpoint is not None) and checkpoint["label"] == "extract_grammar_features":
        ckpt_chunk_size = checkpoint["ckpt_chunk_size"]
        start_i = checkpoint["start_i"]
        results_data = checkpoint["results_data"]
    else:
        ckpt_chunk_size = 100
        # For each ckpt_chunk of 1000 articles, preprocessing input files takes ~20 seconds; GECTOR takes ~400 seconds (on GPU); ~3600 seconds (on CPU).
        start_i = 0
        results_data = {}  # paper_id -> {"abstract": float, "bodytext": float}
        # I need this storing dictionary, since I'm reading the GECTOR results one file by one file (while "abstract" and "bodytext" are written to different files)
    
    while start_i < len(mds):
        # Update checkpoint at start of each ckpt_chunk cycle
        checkpoint = {
            "label": "extract_grammar_features",
            "ckpt_chunk_size": ckpt_chunk_size,
            "start_i": start_i,
            "results_data": results_data
        }
        with open(args.checkpoint, "wb") as f:
            pickle.dump(checkpoint, f)

        end_i = min(start_i + ckpt_chunk_size, len(mds))
        chunk_start_time = time.time()
        os.chdir("/scratch/ssd001/home/zining/pragmatic_academia/scripts/20201109_gector")  # In case the file directory is messed up after checkpointing
        
        # Preprocess input files: Gector takes one sentence per line.
        gector_text_dir = "../gector/data/texts_{}".format(args.slurm_jid)
        gector_input_dir = os.path.join(gector_text_dir, "input")
        gector_output_dir = os.path.join(gector_text_dir, "output")
        if os.path.exists(gector_text_dir):
            shutil.rmtree(gector_text_dir)
            os.makedirs(gector_input_dir)
            os.makedirs(gector_output_dir)
        else:
            os.makedirs(gector_text_dir)
            os.makedirs(gector_input_dir)
            os.makedirs(gector_output_dir)
        for i in range(start_i, end_i):
            md = mds[i]
            paper_id = md['paper_id']
            pdfparse = cat_pdf[paper_id]
            abstract, bt_list = prepare_text(md, pdfparse)
            if abstract is None or md['year'] is None:
                i += 1
                continue 
            abs_list = sent_tokenize(abstract)
            with open(os.path.join(gector_input_dir, f"{paper_id}_abstract"), "w") as f:
                for line in abs_list:
                    f.write(line + "\n")
            with open(os.path.join(gector_input_dir, f"{paper_id}_bodytext"), "w") as f:
                for line in bt_list:
                    f.write(line + "\n")
            # The "_abstract" and "_bodytext" will be processed in ../gector/predict.py; see the predict_for_dir() function there.
    
        # Launch GECTOR
        prev_dir = os.getcwd()  # "scripts/20200830_feature_v2"
        os.chdir("../gector")
        #print ("Switched to {}".format(os.getcwd()))
        tmpfile = tempfile.NamedTemporaryFile(mode="w")
        cmd = """python -u predict.py --model_path models/roberta_1_gector.th --vocab_path data/output_vocabulary --transformer_model roberta --input_dir {} --output_dir {} --log_output {}""".format(gector_input_dir, gector_output_dir, tmpfile.name)
        proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            outs, errs = proc.communicate(timeout=3600)
        except subprocess.TimeoutExpired:
            proc.kill()
            outs, errs = proc.communicate()
        #print("\n =========== gector outputs ===============", outs)
        #print("\n =========== gector errs ===============", errs)
        #print ("\n ===================================")
        os.chdir(prev_dir)
        #print ("Switched back to {}".format(os.getcwd()))

        # Collect results
        gector_result_log = tmpfile.name
        with open(gector_result_log, "r") as f:
            for line in f.readlines():
                linelist = line.split()
                if len(linelist) == 2:
                    paper_id, section = linelist[0].split("_")
                    # paper_id is the string identifier. section is "abstract" or "bodytext"
                    n_err = int(linelist[1])
                    if paper_id in results_data:
                        results_data[paper_id][section] = n_err
                    else:
                        results_data[paper_id] = {section: n_err}
                    
        print ("start_i={}, end_i={}, done in {:.2f} seconds".format(start_i, end_i, time.time() - chunk_start_time))

        # Logistics for this batch: update i
        start_i = end_i
        shutil.rmtree(gector_text_dir)
        
    # Convert into dataframe
    results_df = {"paper_id": [], "grammar_errors_abstract": [], "grammar_errors_bodytext": []}
    for paper_id in results_data:
        if len(results_data[paper_id]) == 2:
            results_df["paper_id"].append(paper_id)
            results_df["grammar_errors_abstract"].append(results_data[paper_id]["abstract"])
            results_df["grammar_errors_bodytext"].append(results_data[paper_id]["bodytext"])
    return pd.DataFrame(results_df)
