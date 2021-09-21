#!/bin/bash
echo "_______________ None"
start=`date +%s`
python src/evaluate_svm.py save_datasets=False load_if_exist=False features.root_distance=False features.is_question=False features.important_words=False features.similarity=False features.polarity=False  features.tfidf=True
end=`date +%s`
echo "$end - $start" | bc -l

echo "_______________ RootDis"
start=`date +%s`
python src/evaluate_svm.py save_datasets=False load_if_exist=False features.root_distance=True features.tfidf=True
end=`date +%s`
echo "$end - $start" | bc -l

echo "_______________ IsQuestion"
start=`date +%s`
python src/evaluate_svm.py save_datasets=False load_if_exist=False features.is_question=True  features.tfidf=True
end=`date +%s`
echo "$end - $start" | bc -l

echo "_______________ HasTwoPart"
start=`date +%s`
python src/evaluate_svm.py save_datasets=False load_if_exist=False features.more_than2_parts=True features.tfidf=True
end=`date +%s`
echo "$end - $start" | bc -l

echo "_______________ Similarity"
start=`date +%s`
python src/evaluate_svm.py save_datasets=False load_if_exist=False  features.similarity=True  features.tfidf=True
end=`date +%s`
echo "$end - $start" | bc -l


echo "_______________ Polarity"
start=`date +%s`
python src/evaluate_svm.py save_datasets=False load_if_exist=False features.polarity=True features.tfidf=True
end=`date +%s`
echo "$end - $start" | bc -l

echo "_______________ ImportantWords"
start=`date +%s`
python src/evaluate_svm.py save_datasets=False load_if_exist=False  features.important_words=True features.tfidf=True
end=`date +%s`
echo "$end - $start" | bc -l


echo "_______________ All"
start=`date +%s`
python src/evaluate_svm.py save_datasets=False load_if_exist=False features.more_than2_parts=True features.root_distance=True features.is_question=True features.important_words=True features.similarity=True  features.polarity=True features.tfidf=True
end=`date +%s`
echo "$end - $start" | bc -l


echo "_______________ All, No root"
start=`date +%s`
python src/evaluate_svm.py save_datasets=False load_if_exist=False features.more_than2_parts=True features.is_question=True features.important_words=True features.similarity=True  features.polarity=True features.tfidf=True
end=`date +%s`
echo "$end - $start" | bc -l