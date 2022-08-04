# stance_detection_persian

The purpose of this project is to find a suitable preprocessing for a Persian corpus; then apply machine learning techniques to detect the stance of a given claim to an arbitrary article. 

## stop-words
There are some meaningless words in the corpus that removing does not make any change in the meaning of a given text. It is worthwhile to choose a suitable stopword list in the dataset context. In different contexts, word values may differ. Four different lists are used to train and test the model separately. To see the effect of each run:

``run /bash/compare_stopwords.sh`` 

## Tokenizer
In the Persian language, some postfix will append into words; For example to modify their ownership. The next step in this project is to use a tokenizer in order to clean the dataset. This step removes unnecessary postfix of words; as the result, the same words will have the same string in the dataset.
Four different methods are compared. Including NLTK, Stanford, Hazm, BERT

``run /bash/compare_tokenizers.sh `` 

## Word Representation
When preprocessings are applied on the dataset, words should represent in a neural network readable format. Three different methods of Bag-of-Words, TFiDF and Word2Vec are compared in this section. 

‍‍‍‍``run /bash/compare_wordrep.sh ``

## oversampling
One way to deal with an imbalanced dataset is to use oversampling methods. There are various methods such as ADASYN, SVMSMOTE, RandomOverSampler, SMOTE and BorderlineSMOTE. 

To apply these methods while training the model run: 

``run /bash/oversampling.sh`` 
