import pandas as pd 
import pickle
import torch 
import numpy as np
import transformers
from nltk.tokenize import sent_tokenize 
from transformers import AutoTokenizer, AutoModelForCausalLM 
import os, sys, time 
from typing import List

from utils import timed_func


def preprocess_for_surprisal(metadata, pdfparse, tokenizer):
    """
    Preprocess one article in this method.
    Returns: abs_data, bt_data
        Both are list of sent pairs (tokenized)
        Is any missing data is encountered, return None for both.
    """
    abstract = metadata['abstract']
    if abstract is None:
        return None, None 
    
    abs_list = [tokenizer.tokenize(sent) for sent in sent_tokenize(abstract)]
    if len(abs_list) == 0:
        return None, None

    bodytext = []
    for section in pdfparse['body_text']:
        if section['text'] is not None:
            bodytext.extend(sent_tokenize(section['text']))
    bt_list = [tokenizer.tokenize(sent) for sent in bodytext]
    if len(bt_list) == 0:
        return None, None
    return abs_list, bt_list


def _compute_surprisals(tokens_data: List[List[str]], tokenizer, model, device) -> List[float]:
    surprisals = []
    for i in range(len(tokens_data) - 1):
        prev = tokens_data[i]  
        subseq = tokens_data[i+1] 
        pos = len(prev)
        

        MAXLEN = tokenizer.model_max_length
        MINLEN_subseq = 10
        MAXLEN_prev = MAXLEN - MINLEN_subseq
        if len(prev) > MAXLEN_prev:
            prev = prev[-MAXLEN_prev:]
        all_tokens = prev + subseq
        if len(all_tokens) > MAXLEN:
            all_tokens = all_tokens[:MAXLEN]
        ids = torch.tensor(tokenizer.convert_tokens_to_ids(all_tokens)).unsqueeze(0).to(device)  # (1, N)
        with torch.no_grad():
            logits, _ = model(ids)
        # logits has shape (batch_size, len, |V|)

        correct_ids = ids[0]
        probs = []  # List of len_subseq, float
        for j in range(len(prev), len(ids[0])):
            probs.append(-logits[0][j][correct_ids[j]].item())
        surprisals.append(np.mean(probs))

    return surprisals 

@timed_func 
def run_surprisal_features_extraction(mds, cat_pdf, args, checkpoint=None, prev_result=None):
    if checkpoint is not None and checkpoint["label"] == "extract_surprisal_features":
        i = checkpoint["i"]
        results = checkpoint["results"]
        prev_result = checkpoint["prev_result"]
    else:
        i = 0
        results = {
            "paper_id": [], 
            "surprisal_abstract_mean": [],
            "surprisal_abstract_std": [],
            "surprisal_bodytext_mean": [],
            "surprisal_bodytext_std": []
        }

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    start_time = time.time()
    while i < len(mds):
        if i % 1000 == 0:
            checkpoint = {
                "label": "extract_surprisal_features",
                "i": i,
                "results": results,
                "prev_result": prev_result
            }
            with open(args.checkpoint, "wb") as f:
                pickle.dump(checkpoint, f)
            time_elapsed = time.time() - start_time
            print ("i={}. Time elapsed: {:.2f} ({:.2f} hrs)".format(
                i, time_elapsed, time_elapsed / 3600
            ))

        md = mds[i]
        paper_id = md['paper_id']
        pdfparse = cat_pdf[paper_id]

        abs_data, bt_data = preprocess_for_surprisal(md, pdfparse, tokenizer)
        if abs_data is None:
            i += 1
            continue 

        abs_surprisals = _compute_surprisals(abs_data, tokenizer, model, device)
        bt_surprisals = _compute_surprisals(bt_data, tokenizer, model, device)

        results['paper_id'].append(paper_id)
        results['surprisal_abstract_mean'].append(np.mean(abs_surprisals))
        results['surprisal_abstract_std'].append(np.std(abs_surprisals))
        results['surprisal_bodytext_mean'].append(np.mean(bt_surprisals))
        results['surprisal_bodytext_std'].append(np.std(bt_surprisals))
        i += 1

    return pd.DataFrame(results), prev_result