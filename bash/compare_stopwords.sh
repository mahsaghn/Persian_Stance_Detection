#!/bin/bash
echo "_______________ no stop word"
start=`date +%s`
python src/evaluate_svm.py features.stopWord_path='/home/mahsa/Desktop/final_project/stance_detection/dataset/stop-words/empty.txt' save_datasets=False load_if_exist=False features.is_question=True features.important_words=True features.similarity=True  features.tfidf=True
end=`date +%s`
echo "$end - $start" | bc -l

echo "_______________ zarharan"
start=`date +%s`
python src/evaluate_svm.py features.stopWord_path='/home/mahsa/Desktop/final_project/stance_detection/dataset/stop-words/stop_words.txt' save_datasets=False load_if_exist=False features.is_question=True features.important_words=True features.similarity=True  features.tfidf=True
end=`date +%s`
echo "$end - $start" | bc -l

echo "_______________ kharazi"
start=`date +%s`
python src/evaluate_svm.py features.stopWord_path='/home/mahsa/Desktop/final_project/stance_detection/dataset/stop-words/nonverbal.txt' save_datasets=False load_if_exist=False features.is_question=True features.important_words=True features.similarity=True  features.tfidf=True
end=`date +%s`
echo "$end - $start" | bc -l 

echo "_______________ shortened"
start=`date +%s`
python src/evaluate_svm.py features.stopWord_path='/home/mahsa/Desktop/final_project/stance_detection/dataset/stop-words/shortened.txt' save_datasets=False load_if_exist=False features.is_question=True features.important_words=True features.similarity=True  features.tfidf=True
end=`date +%s`
echo "$end - $start" | bc -l

echo "_______________ extended"
start=`date +%s`
python src/evaluate_svm.py features.stopWord_path='/home/mahsa/Desktop/final_project/stance_detection/dataset/stop-words/extended.txt' save_datasets=False load_if_exist=False features.is_question=True features.important_words=True features.similarity=True  features.tfidf=True
end=`date +%s`
echo "$end - $start" | bc -l