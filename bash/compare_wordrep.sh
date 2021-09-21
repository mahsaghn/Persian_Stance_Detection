#!/bin/bash
echo "_______________ BOW"
start=`date +%s`
python src/evaluate_svm.py save_datasets=False load_if_exist=False features.is_question=True features.important_words=True features.similarity=True  features.bow=True
end=`date +%s`
echo "$end - $start" | bc -l

echo "_______________ TFIDF"
start=`date +%s`
python src/evaluate_svm.py save_datasets=False load_if_exist=False features.is_question=True features.important_words=True features.similarity=True  features.tfidf=True
end=`date +%s`
echo "$end - $start" | bc -l

echo "_______________ W2V"
start=`date +%s`
python src/evaluate_svm.py save_datasets=False load_if_exist=False features.is_question=True features.important_words=True features.similarity=True  features.w2v=True
end=`date +%s`
echo "$end - $start" | bc -l 
