from dataclasses import dataclass, field
from typing import List

#basepath = '/home/mahsa/Desktop/final_project/stance_detection/'
basepath = '/home/ubuntu/ghaderan/stc/stance_detection/'


@dataclass
class FeatureExtractorConf:
  # important_words: List = field(default_factory=['؟', 'تکذیب',  'تکذیب شد', ':'])
  dataset_path: str = basepath + 'dataset/h2c_new_dataset.csv'
  stopWord_path: str = basepath +  'dataset/stop-words/nonverbal.txt'
  ponctuations_path: str = basepath + 'dataset/ponctuations.txt'
  uniq_claims_path: str = basepath + 'dataset/uniq_claims.txt'
  bert_model_path:str = 'HooshvareLab/bert-fa-zwnj-base'
  stanford_models_path: str = basepath + 'dataset'
  polarity_dataset_path: str = basepath + 'dataset/PerSent.xlsx'
  clean_claims_headlines: str = ''
  clean_claims: str = ''
  clean_headlines: str = ''
  claim_name: str = 'claim'
  headline_name: str = 'headline'
  question_name: str = 'question'
  label_name: str = 'label'
  part_name: str = 'parts'
  w2v_model_path: str = basepath + 'dataset/cc.fa.300.vec'
  save_load_path: str = ''
  save_feature: bool = False
  load_path: bool = save_load_path,
  root_distance: bool = False
  load_if_exist: bool = False
  bow: bool = False
  w2v: bool = False
  tfidf: bool = True
  polarity: bool = True
  similarity: bool = True
  important_words: bool = True
  is_question: bool = True
  more_than2_parts: bool = True
  tokenize_method: str = 'hazm'#nltk, stanford, bert, hazm

@dataclass
class SVM:
  C: float =  2.5
  coef0 : float = 0.0
  tol : float = 1e-3
  kernel : str = 'rbf' #‘poly’, ‘rbf’, ‘sigmoid’
  degree: int = 2 # degree of poly,C=3.5

@dataclass
class LinearSVC:
  C: float =  2.5
  coef0 : float = 0.0
  tol : float = 0.0001
  penalty: str = 'l2'
  loss: str = 'hinge'

@dataclass
class RandomForest:
  criterion : str = 'gini'
  max_features : str = 'None'
  n_estimators : int = 125

@dataclass
class LogisticRegression:
  C: float =  3.0
  solver: str = 'saga'

@dataclass
class H2CBaselineConfig:
  save_path: str = '/models/'
  save_datasets:bool = False
  load_path: str = basepath + 'dataset/seperated'
  load_if_exist: bool = False
  test_size: float = 0.2
  over_sample: bool = False
  oversampling: str = 'ADASYN'
  N_neighbors: int = 9
  Random_state: int = 88
  features: FeatureExtractorConf = FeatureExtractorConf()
  svc : SVM = SVM()
  linear_svc : LinearSVC = LinearSVC()
  logistic_regression : LogisticRegression = LogisticRegression()
  random_forest : RandomForest = RandomForest()
