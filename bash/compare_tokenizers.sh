#!/bin/bash

echo "_______________ NLTK"
python src/save_tokenized.py features.tokenize_method='nltk'

echo "_______________ Stanford"
python src/save_tokenized.py features.tokenize_method='stanford'

echo "_______________ Hazm"
python src/save_tokenized.py features.tokenize_method='hazm'

#echo "_______________ Bert"
#python src/save_tokenized.py features.tokenize_method='bert'
