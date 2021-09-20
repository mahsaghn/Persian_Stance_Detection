#!/bin/bash

echo "pwd: `pwd`"


echo "_______________TFIDF"
python src/oversampling_baseline.py  oversampling=False save_datasets=True load_if_exist=True features.is_question=True features.important_words=True features.similarity=True  features.tfidf=True
python src/oversampling_baseline.py  oversampling=True save_datasets=True load_if_exist=True features.is_question=True features.important_words=True features.similarity=True  features.tfidf=True

echo "_______________BOW"
python src/oversampling_baseline.py  oversampling=False save_datasets=True load_if_exist=True features.is_question=True features.important_words=True features.similarity=True features.bow=True
python src/oversampling_baseline.py  oversampling=True save_datasets=True load_if_exist=True features.is_question=True features.important_words=True features.similarity=True  features.tfidf=True

echo "_______________W2V"
python src/oversampling_baseline.py  oversampling=False save_datasets=True load_if_exist=True features.is_question=True features.important_words=True features.similarity=True features.w2v=True
python src/oversampling_baseline.py  oversampling=True save_datasets=True load_if_exist=True features.is_question=True features.important_words=True features.similarity=True  features.tfidf=True
