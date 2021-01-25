import re
import pandas as pd
import numpy as np
from sklearn import model_selection, preprocessing, linear_model
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import decomposition, ensemble
import nltk
import nltk.corpus
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle

trainDF=pd.read_csv('clean.csv',encoding='mac_roman', na_filter=True, na_values='[]')
trainDF.dropna(inplace=True)


X=trainDF['text_tokens_lemma']
Y=trainDF['Labels']
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X, Y,test_size=0.2)

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text_tokens_lemma'])
pickle.dump(count_vect, open('cv1.pkl','wb'))
# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)



loreg=linear_model.LogisticRegression(max_iter=200)# initialize the model
loreg.fit(xtrain_count, train_y) # fit he model
pickle.dump(loreg, open('model1.pkl','wb'))
y_pred=loreg.predict(xvalid_count) # now predict
cm = confusion_matrix(valid_y, y_pred)   


cv = pickle.load(open('cv1.pkl','rb'))
model = pickle.load(open('model1.pkl','rb'))
wordnet = WordNetLemmatizer()

review=input("Enter a tweet:")
corpus = []
review = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", '', review)
review = re.sub(r"\d+", '', review)
review = review.lower()
review= nltk.word_tokenize(review)
review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
review = ' '.join(review)
corpus.append(review)
corpus                       
print(corpus)

corpus = cv.transform(corpus)
pred=model.predict(corpus)
if(pred==0):
    print("Human Tweeted")
else:
    print("Bot Tweeted")









