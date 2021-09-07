from dataclasses import dataclass, field
from typing import List

@dataclass
class FeatureExtractorConf:
  # important_words: List = field(default_factory=['؟', 'تکذیب',  'تکذیب شد', ':'])
  dataset_path: str = '/home/mahsa/Desktop/final_project/stance_detection/dataset/h2c_new_dataset.csv'
  stopWord_path: str = '/home/mahsa/Desktop/final_project/stance_detection/dataset/stop_words.txt'
  ponctuations_path: str = '/home/mahsa/Desktop/final_project/stance_detection/dataset/ponctuations.txt'
  uniq_claims_path: str = '/home/mahsa/Desktop/final_project/stance_detection/dataset/uniq_claims.txt'
  bert_model_path:str = 'HooshvareLab/bert-base-parsbert-uncased'
  polarity_dataset_path: str = ''
  stanford_models_path: str = ''
  clean_claims_headlines: str = ''
  clean_claims: str = ''
  clean_headlines: str = ''
  claim_name: str = 'claim'
  headline_name: str = 'headline'
  question_name: str = 'question'
  label_name: str = 'label'
  part_name: str = 'parts'
  w2v_model_path: str = ''
  save_load_path: str = ''
  save_feature: bool = False
  load_path: bool = save_load_path,
  root_distance: bool = False
  load_if_exist: bool = False
  bow: bool = False
  w2v: bool = False
  polarity: bool = False
  tfidf: bool = False
  similarity: bool = False
  important_words: bool = False
  is_question: bool = False
  more_than2_parts: bool = False

@dataclass
class H2CBaselineConfig:
  oversampling: str = 'ADASYN'
  N_neighbors: int = 9
  Random_state: int = 88
  features: FeatureExtractorConf = FeatureExtractorConf()