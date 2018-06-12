import numpy as np
import pandas as pd
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense ,Dropout,Activation,Embedding
from keras.layers import LSTM


train_df = pd.read_csv('data/train.csv',header = 0)
test_df = pd.read_csv('data/test.csv',header = 0)
print(train_df.head())
print(test_df.head())

raw_doc_train = train_df['tweet'].values
raw_doc_test = test_df['tweet'].values
positive_train = train_df['positive']
num_labels = len(np.unique(positive_train))

print('text pre-processing ')
stop_words = set(stopwords.words('english'))
stop_words.update(['.',',','"',':',';','(',')','[',']','{','}'])
stemmer = SnowballStemmer('english')

print('pre-processing train docs...')
processed_docs_train =[]
for doc in raw_doc_train:
    tokens = word_tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    stemmed = [stemmer.stem(word) for word in filtered]
    processed_docs_train.append(stemmed)

print('pre-processing test docs...')
processed_docs_test = []
for doc in raw_doc_test:
    tokens = word_tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    stemmed = [stemmer.stem(word) for word in filtered]
    processed_docs_test.append(stemmed)


processed_docs_all = np.concatenate((processed_docs_train,processed_docs_test))
dictionary = corpora.Dictionary(processed_docs_all)
dictionary_size = len(dictionary.keys())
print('dictionary:',dictionary_size)

print('converting to token ids...')
word_id_train ,word_id_len =[],[]
for doc in processed_docs_train:
    word_ids = [dictionary.token2id[word] for word in doc]
    word_id_train.append(word_ids)
    word_id_len.append(len(word_ids))

word_id_test,word_ids = [],[]
for doc in processed_docs_test:
    word_ids = [dictionary.token2id[word] for word in doc]
    word_id_test.append(word_ids)
    word_id_len.append(len(word_ids))

seq_len = np.round((np.mean(word_id_len)+2*np.std(word_id_len))).astype(int)
word_id_train = sequence.pad_sequences(np.array(word_id_train),maxlen =seq_len)
word_id_test = sequence.pad_sequences(np.array(word_id_test),maxlen = seq_len)
y_train_enc = np_utils.to_categorical(positive_train, num_labels)

print('LSTM')
model = Sequential()
model.add(Embedding(dictionary_size,128,dropout = 0.5))
model.add(LSTM(128,dropout_W = 0.2,dropout_U=0.2))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer = 'adam',metrics=['accuracy'])
model.fit(word_id_train,y_train_enc,nb_epoch=1,batch_size =256,verbose=1)

test_pred = model.predict_classes(word_id_test)
test_df['positive'] = test_pred.reshape(-1,1)
header = ['id','positive']
test_df.to_csv('/Users/martin_yan/Desktop/submission.csv',columns=header,index = False,header = True)

