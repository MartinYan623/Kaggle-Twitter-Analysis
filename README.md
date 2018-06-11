# Kaggle-Twitter-Analysis

How to fix the problem relating 'Resource punkt not found'.

Firstly, enter a python shell and then input following sentences sequentially:

>>> import nltk

>>> nltk.download()

Then, we can see a windows appear now, and select 'models' sub-windows.

Next, we choose 'punkt' package to download.

After this, finally it works!


Here is a log about this project on kaggle competition called twitter analysis.

2018.6.4

(1) Simply use textblob to predict the sentiment of sentences even do not use train data set.

(2) Install nltk, but encounter some problems. After searching methods online, I have fixed it already.

(3) NaiveBayesClassifier for large train data set, the speed of training is very low.

Got score 0.62329 (more higher, more better)

2018.6.5

(1) Use nltk built in sentiment analysis as follows:

#from nltk.sentiment.vader import SentimentIntensityAnalyzer

This is also a method which we do not use train data set and directly predict the sentiment condition of test data set.

Got score 0.65036

2018.6.11

(1) Use nltk again to download the 'stopwords'. nltk is a tool for natural language processing.

(2) Lstm method is used to do sentiment analysis and get a much improvement.

the core code as follows:

#model = Sequential()

#model.add(Embedding(dictionary_size,128,dropout = 0.2))

#model.add(LSTM(128,dropout_W = 0.2,dropout_U=0.2))

#model.add(Dense(num_labels))

#model.add(Activation('softmax'))

#model.compile(loss='categorical_crossentropy',optimizer = 'adam',metrics=['accuracy'])

#model.fit(word_id_train,y_train_enc,nb_epoch=1,batch_size =256,verbose=1)

consider to adjust the parameter of lstm and hope to get better result.

Got score 0.79445

