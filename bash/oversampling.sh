#!/bin/bash

echo "pwd: `pwd`"


echo "_______________adasyn"
python src/main.py  over_sample=True oversampling='ADASYN'
python src/main.py  over_sample=True oversampling='ADASYN'  features.dataset_path='/home/ubuntu/ghaderan/stc/stance_detection/dataset/stance+fever/h2c.csv'

echo "_______________SVMSMOTE"
python src/main.py  over_sample=True oversampling='SVMSMOTE'
python src/main.py  over_sample=True oversampling='SVMSMOTE' features.dataset_path='/home/ubuntu/ghaderan/stc/stance_detection/dataset/stance+fever/h2c.csv'

echo "_______________RandomOverSampler"
python src/main.py  over_sample=True oversampling='RandomOverSampler'
python src/main.py  over_sample=True oversampling='RandomOverSampler' features.dataset_path='/home/ubuntu/ghaderan/stc/stance_detection/dataset/stance+fever/h2c.csv'

echo "_______________SMOTE"
python src/main.py  over_sample=True oversampling='SMOTE'
python src/main.py  over_sample=True oversampling='SMOTE' features.dataset_path='/home/ubuntu/ghaderan/stc/stance_detection/dataset/stance+fever/h2c.csv'


echo "_______________BorderlineSMOTE"
python src/main.py  over_sample=True oversampling='BorderlineSMOTE'
python src/main.py  over_sample=True oversampling='BorderlineSMOTE' features.dataset_path='/home/ubuntu/ghaderan/stc/stance_detection/dataset/stance+fever/h2c.csv'

