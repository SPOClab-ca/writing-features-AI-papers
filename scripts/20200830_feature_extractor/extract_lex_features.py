import argparse
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import cmudict
prondict = cmudict.dict()
import numpy as np
import json
from lexicalrichness import LexicalRichness
import os, sys, time
import pandas as pd
import pickle

from utils import timed_func


def extract_basic_info(metadata, pdfparse):
    curr_year = 2021
    citation_per_year = len(metadata['inbound_citations']) / (curr_year - metadata['year'])
    
    basic_info = {
        "paper_id": [metadata['paper_id']],
        "n_citations": [len(metadata['inbound_citations'])],
        "annual_citations": [citation_per_year],
        "n_author": [len(metadata['authors'])],
        "n_outbound_citations": [len(metadata["outbound_citations"])],
        "title_word_length": [len(metadata['title'].split())],
        "num_sections": [len(pdfparse['body_text'])],
    }
    basic_info = pd.DataFrame(basic_info)
    return basic_info

def extract_doc_lens(abstract, bodytext):
    bt_word_count = 0
    bt_sent_count = 0
    for bt in bodytext:
        bt_word_count += len(bt.split())
        bt_sent_count += len(sent_tokenize(bt))
    doc_len_info = {
        "abstract_word_counts": [len(abstract.split())],
        "abstract_sent_counts": [len(sent_tokenize(abstract))],
        "bodytext_word_counts": [bt_word_count],
        "bodytext_sent_counts": [bt_sent_count]
    }
    doc_len_info = pd.DataFrame(doc_len_info)
    return doc_len_info

def extract_sent_lens(abstract, bodytext):
    abs_sents = sent_tokenize(abstract)
    abs_lens = [len(s.split()) for s in abs_sents]
    
    bt_lens = [len(s.split()) for s in bodytext]
    
    sent_lens_info = {
        "sent_lens_abs_mean": [np.mean(abs_lens)],
        "sent_lens_abs_var": [np.var(abs_lens)],
        "sent_lens_bodytext_mean": [np.mean(bt_lens)],
        "sent_lens_bodytext_var": [np.var(bt_lens)]
    }
    return pd.DataFrame(sent_lens_info)

def extract_mattr(abstract, bodytext):
    lex_rich_info = {}
    lengths = [5, 10, 20, 30, 40]
    lex = LexicalRichness(abstract)
    if lex.words == 0:
        return None
    for length in lengths:
        name = f"lex_mattr_{length}_abstract"
        if lex.words < length:
            lex_rich_info[name] = [lex.ttr]
        else:
            lex_rich_info[name] = [lex.mattr(length)]
    bt = "\n".join(bodytext)
    lex = LexicalRichness(bt)
    if lex.words == 0:
        return None
    for length in lengths:
        name = f"lex_mattr_{length}_bodytext"
        if lex.words < length:
            lex_rich_info[name] = [lex.ttr] 
        else:
            lex_rich_info[name] = [lex.mattr(length)]

    return pd.DataFrame(lex_rich_info)

numsyllables_pronlist = lambda l: len([s for s in l if lambda s: isdigit(s.encode('ascii', 'ignore').lower()[-1])])
not_punctuation = lambda w: not (len(w)==1 and (not w.isalpha()))
get_word_count = lambda text: len([w for w in word_tokenize(text) if not_punctuation(w)])
get_sent_count = lambda text: len(sent_tokenize(text))

def numsyllables(word):
    try:
        return list(set(map(numsyllables_pronlist, prondict[word.lower()])))
    except KeyError:
        return [0]

def extract_readability(abstract, bodytext):
    """
    See https://datawarrior.wordpress.com/2016/03/29/flesch-kincaid-readability-measure/
    """
    def _flesch_formula(word_count, sent_count, syllable_count):
        return 206.835 - 1.015*word_count/sent_count - 84.6*syllable_count/word_count

    def _flesch_kincaid(word_count, sent_count, syllable_count):
        return 0.39 * word_count / sent_count + 11.8 * syllable_count / word_count - 15.59

    def _text_statistics(text):
        word_count = get_word_count(text)
        sent_count = get_sent_count(text)
        syllable_count = sum(map(lambda w: max(numsyllables(w)), word_tokenize(text)))
        return word_count, sent_count, syllable_count
    
    wc_abs, sc_abs, syc_abs = _text_statistics(abstract)
    bt = "\n".join(bodytext)
    wc_bt, sc_bt, syc_bt = _text_statistics(bt)
    if min(wc_abs, sc_abs, syc_abs, wc_bt, sc_bt, syc_bt) == 0:
        return None  # Prevents DivideByZero

    readability_info = {
        "flesch_read_ease_abstract": [_flesch_formula(wc_abs, sc_abs, syc_abs)], 
        "flesch_read_ease_bodytext": [_flesch_formula(wc_bt, sc_bt, syc_bt)], 
        "flesch_kincaid_grade_level_abstract": [_flesch_kincaid(wc_abs, sc_abs, syc_abs)],
        "flesch_kincaid_grade_level_bodytext": [_flesch_kincaid(wc_bt, sc_bt, syc_bt)]
    }
    return pd.DataFrame(readability_info)


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
def run_lex_features_extraction(mds, cat_pdf, args, print_per=10000, checkpoint=None, prev_result=None):
    
    if checkpoint is not None and checkpoint["label"] == "extract_lex_features":
        all_dfs = checkpoint["all_dfs"]
        skipped = checkpoint["skipped"]
        start_time = checkpoint["start_time"]
        i = checkpoint["i"]
        prev_result = checkpoint["prev_result"]    
    else:
        all_dfs = []
        skipped = 0
        start_time = time.time()
        i = 0
    while i < len(mds):
        md = mds[i]
        if (i+1) % print_per == 0:
            print ("Iterating row {}. skipped={}. Time slapsed={:.2f} hours".format(i+1, skipped, (time.time() - start_time) / 3600))
            checkpoint = {
                "label": "extract_lex_features",
                "all_dfs": all_dfs,
                "skipped": skipped,
                "start_time": start_time,
                "i": i,
                "prev_result": prev_result}
            with open(args.checkpoint, "wb") as f:
                pickle.dump(checkpoint, f)
                
        pdfparse = cat_pdf[md['paper_id']]
        
        abstract, bodytext = prepare_text(md, pdfparse)
        if abstract is None or md['year'] is None:
            skipped += 1
            i += 1
            continue
            
        basic_info = extract_basic_info(md, pdfparse)
        doclen_info = extract_doc_lens(abstract, bodytext)
        sent_lens = extract_sent_lens(abstract, bodytext)
        mattr = extract_mattr(abstract, bodytext)
        if mattr is None:  # Even if prepare_text handled empty contents, mattr might still throw exception here.
            skipped += 1
            i += 1
            continue 
        readability_info = extract_readability(abstract, bodytext)
        feat_df = pd.concat([
            basic_info, 
            doclen_info, 
            sent_lens, 
            mattr, 
            readability_info], axis=1)
        all_dfs.append(feat_df)
        i += 1
        
    feat_df = pd.concat(all_dfs, ignore_index=True, sort=False)
    print ("Num. features: {}".format(len(list(feat_df))))
    print ("Collected {} items. Skipped {}".format(
        len(feat_df), skipped
    ))
    return feat_df, prev_result
