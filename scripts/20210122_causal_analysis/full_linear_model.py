import argparse
import dowhy
from dowhy import CausalModel
import numpy as np
from pathlib import Path
import pandas as pd
import os, sys, time
import scipy
from tqdm import tqdm

from utils import timed_func, get_venue_names


def build_gml_string(features_df, target, treatment):
    node_str = ""
    edge_str = ""
    confounder_id = 0
    for ft in features_df.columns:
        if ft == treatment or ft == target: 
            continue
        node_str += 'node[id "C{}" label "{}"]\n'.format(confounder_id, ft)
        edge_str += '\nedge[source "C{}" target "Y"]'.format(confounder_id)
        confounder_id += 1
        
    gml_string = """graph[directed 1 node[id "Y" label "{}"]
        node[id "X" label "{}"]
        {}edge[source "X" target "Y"]{}]""".format(target, treatment, node_str, edge_str)
    return gml_string


def causal_inference(features_df, target='venue_is_top'):
    report_df = {'feature': [], 'causal_estimate': [], 'p_value': [], 'spearman_r': [], 'spearman_p': []}
    for ft in tqdm(features_df.columns):
        if ft == target: continue
        print ("\nFeature:", ft)
        
        # Model
        gml_string = build_gml_string(features_df, target, treatment=ft)
        model = CausalModel(data=features_df, 
                            treatment=ft,
                           outcome=target, 
                           graph=gml_string)
        
        # Identify estimand
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        
        # Estimation
        causal_estimate_reg = model.estimate_effect(
            identified_estimand, method_name="backdoor.linear_regression", 
            test_significance=True)
        causal_estimate_val = causal_estimate_reg.value 
        p_value = causal_estimate_reg.test_stat_significance()['p_value'][0]

        report_df['feature'].append(ft)
        report_df['causal_estimate'].append(causal_estimate_val)
        report_df['p_value'].append(p_value)

        # Also compute Spearman Correlation
        rho, pval = scipy.stats.spearmanr(features_df[ft], features_df[target])
        report_df['spearman_r'].append(rho)
        report_df['spearman_p'].append(pval)
        
    causal_report = pd.DataFrame(report_df)
    causal_report['significant'] = causal_report.p_value < 0.05
    return causal_report.sort_values(by="causal_estimate", ascending=False)



venue_names = [
    'ACL', 'ACL_v_arxiv', 'NAACL', 'NAACL_v_arxiv',
    'EMNLP', 'EMNLP_v_arxiv', 'COLING', 'COLING_v_arxiv',
    'NIPS', 'NIPS_v_arxiv', 'ICML', 'ICML_v_arxiv',
    'AAAI', 'AAAI_v_arxiv', 'IJCAI', 'IJCAI_v_arxiv',
    'ICRA', 'ICRA_v_arxiv', 'CVPR', 'CVPR_v_arxiv', 'ICASSP'
]
category_names = ['NLP', 'ML', 'AI', 'CV', 'Robo', 'Speech']


@timed_func 
def main(features, args):
    """
    Takes about 20 mins to run on ACL (85 features, 2000 samples). 
    """
    if args.target == "venue_is_top":
        results_path = "classification_results"
        features = features.drop(columns=['paper_id', 'n_citations', 'annual_citations'])
    else:
        results_path = "regression_results"
        features = features.drop(columns=['paper_id', 'n_citations', 'venue_is_top'])
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    if args.by_venue:
        venue_name = venue_names[args.venue_name_id]
        df = features[features.venue.isin(get_venue_names(venue_name))].drop(columns=['venue', 'venue_category'])
        causal_report = causal_inference(df, args.target)
        causal_report.to_csv(Path(results_path, venue_name+".csv"), index=False)    
    elif args.by_category: 
        cat_name = category_names[args.cat_id]
        df = features[features.venue_category == cat_name].drop(columns=['venue', 'venue_category'])
        causal_report = causal_inference(df, args.target)
        causal_report.to_csv(Path(results_path, cat_name+".csv"), index=False)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--by_venue", action="store_true")
    parser.add_argument("--venue_name_id", type=int, default=0, choices=range(len(venue_names)))
    parser.add_argument("--by_category", action="store_true")
    parser.add_argument("--cat_id", type=int, default=0, choices=range(len(category_names)))

    parser.add_argument("--target", type=str, choices=['annual_citations', 'venue_is_top'])
    parser.add_argument("--remove_redundant_features", action="store_true")

    args = parser.parse_args()
    print(args)
    features = pd.read_csv("../../data/features_v2_with_venue.csv")
    if args.remove_redundant_features:
        features = features.drop(columns=[
            "num_sections", "bodytext_word_counts", "bodytext_sent_counts",  # Remove article length features
            "lex_mattr_5_abstract", "lex_mattr_20_abstract", "lex_mattr_30_abstract", "lex_mattr_40_abstract",  # Only keep MATTR_10
            "lex_mattr_5_bodytext", "lex_mattr_20_bodytext", "lex_mattr_30_bodytext", "lex_mattr_40_bodytext"
        ])
    main(features, args)
