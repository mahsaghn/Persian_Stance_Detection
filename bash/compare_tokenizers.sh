#!/bin/bash

echo "_______________ NLTK"
start=`date +%s`
#python src/save_tokenized.py features.tokenize_method='nltk'
end=`date +%s`
echo "$end - $start" | bc -l 

echo "_______________ Stanford"
start=`date +%s`
python src/save_tokenized.py features.tokenize_method='stanford'
end=`date +%s`
echo "$end - $start" | bc -l 

echo "_______________ Hazm"
start=`date +%s`
#python src/save_tokenized.py features.tokenize_method='hazm'
end=`date +%s`
echo "$end - $start" | bc -l 

echo "_______________ Bert"
start=`date +%s`
#python src/save_tokenized.py features.tokenize_method='bert'
end=`date +%s`
echo "$end - $start" | bc -l 





