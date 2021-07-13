import itertools
import pickle 
import numpy as np 
from pathlib import Path 
import sklearn 
from sklearn.neural_network import MLPRegressor 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import timed_func

@timed_func 
def process_vectorize(cid_start, cid_end, abstract=True, bodytext=True, max_features=1000):
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=max_features)
    texts = []
    labels = []
    for cid in range(cid_start, cid_end):
        with open(f"../../data/predict_citations_text_only/chunk_{cid}.pkl", "rb") as f:
            articles = pickle.load(f)
        for article in articles:
            article_text = []
            if abstract:
                article_text.append(article['abstract'])
            if bodytext:
                article_text.extend(article['bodytext'])
            texts.append(" ".join(article_text))
        
        labels.append(article['annual_citations'])
    X = vectorizer.fit_transform(texts)
    Y = np.array(labels)
    return X, Y, vectorizer.get_feature_names()


def regression_select_models(XYF, models, verbose=True):
    """
    Same training scheme as 20210423_classify_select_features.ipynb
    """
    X, Y, feature_names = XYF
    
    skf = StratifiedKFold(n_splits=6)
    importances = []
    fold_accs = []
    fold_f1_scores = []
    best_model_names = []
    for trval_idx, test_idx in skf.split(X, Y):
        # Sweep through models in these folds. Choose the best one. Classify
        X_train, X_dev, Y_train, Y_dev = train_test_split(
            X[trval_idx], Y[trval_idx], test_size=0.2, stratify=Y[trval_idx]
        )
        X_test, Y_test = X[test_idx], Y[test_idx]
        f1_scores = []
        trained_models = []
        for model_name in models:
            model = sklearn.base.clone(models[model_name])
            try:
                model.fit(X_train, Y_train)
                Y_dev_pred = model.predict(X_dev)
                f1_scores.append(f1_score(Y_dev, Y_dev_pred))
            except ValueError:
                f1_scores.append(0)
            trained_models.append(model)
        
        max_id = np.argmax(f1_scores)
        model = trained_models[max_id]
        Y_test_pred = model.predict(X_test)
        fold_f1_scores.append(f1_score(Y_test, Y_test_pred))
        fold_accs.append(accuracy_score(Y_test, Y_test_pred))
    
        best_model_name = list(models.keys())[max_id]
        best_model_names.append(best_model_name)
    
        # Select the most important features
        selector = SelectFromModel(model)
        selector.fit(X[trval_idx], Y[trval_idx])
        if hasattr(model, "coef_"):
            importances.append(np.absolute(model.coef_[0]))
        elif hasattr(model, "feature_importances_"):
            importances.append(model.feature_importances_)
        else:
            pass  # Model doesn't support selecting features
        
    acc_mean, acc_std, f1_mean, f1_std = np.mean(fold_accs), np.std(fold_accs), np.mean(fold_f1_scores), np.std(fold_f1_scores)
    if len(importances) > 0:
        mean_imp = np.mean(importances, axis=0)
        top_features = np.array(feature_names)[np.argsort(-mean_imp)]
        top_features_str = "Top5 feats: " + "; ".join(top_features[:5])
    else:
        top_features = None
        top_features_str = ""
    
    if verbose:
        print("Acc: mean {:.4f}, std {:.4f}; F1: mean {:.4f}, std {:.4f}".format(
            acc_mean, acc_std, f1_mean, f1_std))
        print(top_features_str)
        
    fold_f1_scores_str = ", ".join(["{:.4f}".format(fs) for fs in fold_f1_scores])
    bm_str = ", ".join(best_model_names)
    return fold_f1_scores_str, top_features_str, bm_str


@timed_func
def select_features_sweep_params(XYF, verbose=True):
    models = OrderedDict({        
        # Models without support to selecting features:
        "mlp_10": MLPRegressor([10]),
        "mlp_20": MLPRegressor([20]),
        "mlp_40": MLPRegressor([40]),
        "mlp_80": MLPRegressor([80]),
        "mlp_10_10": MLPRegressor([10,10]),
        "mlp_20_20": MLPRegressor([20,20]),
        "mlp_20_20": MLPRegressor([40,40])
    })
    
    return classify_select_models(XYF, models, verbose)


class MyIterator:
    def __init__(self, min_cid=0, max_cid=99):
        self.articles = []
        self.i = 0
        self.min_cid = min_cid 
        self.max_cid = max_cid
        self.next_cid = 0
    
    def reset(self):
        self.i = 0
        self.next_cid = self.min_cid
        self.articles = []

    def __iter__(self):
        return self 

    def __next__(self):
        self.i += 1
        if self.i >= len(self.articles):
            if self.next_cid >= self.max_cid:
                self.reset()
                raise StopIteration
            else:
                with open(f"../../data/predict_citations_text_only/chunk_{self.next_cid}.pkl", "rb") as f:
                    self.articles = pickle.load(f)
            self.i = 0 
            self.next_cid += 1
              
        return self.articles[self.i]
        

if __name__ == "__main__":
    XYF = process_vectorize(cid_start=91, cid_end=100, abstract=True, bodytext=True, max_features=100)
    with open("tfidf_xyf_ab100_chunks91_100.pkl", "wb") as f:
        pickle.dump(XYF, f)
    
    #with open("tfidf_xyf_ab100_chunks0_10.pkl", "rb") as f:
    #    XYF = pickle.load(f)
    #f1_scores, top_features_str, bm_str = select_features_sweep_params(XYF)
    #print(f1_scores)
    #print(bm_str)
