import pandas as pd
from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
import stanfordnlp
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest

import os.path
import joblib

from feature_extractor import PSFeatureExtractor as FeatureExtractor

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import (SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC)
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import hydra
from config.baseline_h2c import H2CBaselineConfig, FeatureExtractorConf
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="config", node=H2CBaselineConfig)
cs.store(group="",name="baseline_h2c", node=H2CBaselineConfig)
cs.store(group="features",name='features',node=FeatureExtractorConf)
@hydra.main(config_path="config", config_name="config")

def main(cfg: H2CBaselineConfig):
    psf_extractor = FeatureExtractor(cfg=cfg.features)
    tokens_claims, tokens_headlines = psf_extractor.nltk_tokenize()
    features_tfidf, features_name_tfidf = psf_extractor.generate_Features()

    print("done")

main()

#stanford_models_path  = '/content/drive/My Drive/persian_stance_baseline_data'
# dataset_path = '/content/drive/My Drive/multilingual_bert_dataset/dataset/HeadlineToClaim_multilingualBERT.csv'
# polarity_dataset_path = '/content/drive/My Drive/persian_stance_baseline_data/dataset/PerSent.xlsx'
# save_load_path = "/content/drive/My Drive/persian_stance_baseline_data/vectors"
# w2v_model_path = "/content/drive/My Drive/persian_stance_baseline_data/vectors/w2v_persian.pkl"
# train_test_sets_save_path = "/content/drive/My Drive/persian_stance_baseline_data/dataset"
# train_test_sets_load_path = "/content/drive/My Drive/persian_stance_baseline_data/dataset"
