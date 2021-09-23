from dataclasses import dataclass, field
from typing import List
from baseline_h2c  import FeatureExtractorConf

# basepath = '/home/mahsa/Desktop/final_project/stance_detection/'
basepath = '/home/ubuntu/ghaderan/stc/stance_detection/'



@dataclass
class H2CDeepConfig:
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
  model_path: str = 'HooshvareLab/bert-fa-zwnj-base'
  checkpoint_path : str = basepath + 'models/h2c_parsbert_weights_newdataset.h5'
  lr : float = 0.000035
  batch_size: int = 128
  epochs : int = 8
  MAX_LEN : int = 32
