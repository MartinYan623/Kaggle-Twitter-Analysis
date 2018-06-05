import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import numpy as np
import nltk
from textblob.classifiers import NaiveBayesClassifier

"""
# Simply use textblob to predict the sentiment of sentences
prediction=[]
for i in range(len(ta_test)):
    text = ta_test.iloc[i][1]
    blob = TextBlob(text)
    score=blob.sentiment.polarity
    if score>0:
        prediction.append(1)
    else:
        prediction.append(0)

submission = pd.DataFrame({
        "id":ta_test['id'],
        "prediction": prediction
    })
print(submission)
submission.to_csv('/Users/martin_yan/Desktop/submission.csv', index=False)

# ntlk built in sentiment analysis method
#nltk.download('vader_lexicon')
ta_train = pd.read_csv('data/train.csv')
ta_test = pd.read_csv("data/test.csv")
prediction=[]
id=[]
sid = SentimentIntensityAnalyzer()
for index,row in ta_test.iterrows():
    print(index)
    id.append(row['id'])
    ss = sid.polarity_scores(row['tweet'])
    neg=ss['neg']
    pos=ss['pos']
    if pos>neg:
        prediction.append(1)

    else:
        prediction.append(0)

submission = pd.DataFrame({
        "id":ta_test['id'],
        "prediction": prediction
    })
print(submission)
submission.to_csv('/Users/martin_yan/Desktop/submission.csv', index=False)
"""

# nltk朴素贝叶斯分类判断 数据量太大很慢
ta_train = pd.read_csv('data/train.csv')
ta_test = pd.read_csv("data/test.csv")

def doc_feature(doc):
    train = []
    for index, row in doc.iterrows():
        tuple = (row['tweet'],row['positiev'])
        train.append(tuple)
    return train
train1=nltk.apply_features(doc_feature,ta_train)
cl = NaiveBayesClassifier(train1)
print(cl.classify("Their burgers are amazing"))


