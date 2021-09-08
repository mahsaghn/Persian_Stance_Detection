#!/bin/bash

echo "pwd: `pwd`"


echo "_______________TFIDF"
python src/oversampling_baseline.py  save_datasets=True load_if_exist=True features.is_question=True features.important_words=True features.similarity=True  features.tfidf=True 

echo "_______________BOW"
python src/oversampling_baseline.py  save_datasets=True load_if_exist=True features.is_question=True features.important_words=True features.similarity=True features.bow=True

echo "_______________W2V"
python src/oversampling_baseline.py  save_datasets=True load_if_exist=True features.is_question=True features.important_words=True features.similarity=True features.w2v=True
