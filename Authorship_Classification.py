# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 17:39:39 2019

@author: Kirti Sodhi
"""
############ IMPORTING Libraries 
import nltk, re, pprint
from nltk import word_tokenize
from urllib import request
from nltk import sent_tokenize
from nltk.corpus import stopwords
import numpy as np
import random

####### Importing First Book three Expeditions into the Interior of Eastern Australia
import requests
response = requests.get('http://www.gutenberg.org/cache/epub/13033/pg13033.txt')
text=response.text
text=text.rstrip()
text=text.lower()
tokenized_text=sent_tokenize(text)
tokenized_word=word_tokenize(text)
stop_words=set(stopwords.words("english"))
filtered_sent=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered_sent.append(w)

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()
stemmed_words=[]
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))
words = [word for word in stemmed_words if word.isalpha()]
x1=[]
for i in range (0,200*150,150):
     x1.append(words[i:i+150])
a=[0]*200


######################## Importing second book Dracula
import requests
response1 = requests.get('http://www.gutenberg.org/cache/epub/345/pg345.txt')
text1=response1.text
text1=text1.rstrip()
text1=text1.lower()
tokenized_text1=sent_tokenize(text1)
tokenized_word1=word_tokenize(text1)
stop_words1=set(stopwords.words("english"))

filtered_sent1=[]
for w in tokenized_word1:
    if w not in stop_words1:
        filtered_sent1.append(w)

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()
stemmed_words1=[]
for w in filtered_sent1:
    stemmed_words1.append(ps.stem(w))

words1 = [word for word in stemmed_words1 if word.isalpha()]
x2=[]
for i in range (0,200*150,150):
     x2.append(words1[i:i+150])
b=[1]*200


############## Importing third book Sandwiches
import requests
response2 = requests.get('http://www.gutenberg.org/cache/epub/29329/pg29329.txt')
text2=response2.text
text2=text2.rstrip()
text2=text2.lower()
tokenized_text2=sent_tokenize(text2)
tokenized_word2=word_tokenize(text2)

stop_words2=set(stopwords.words("english"))
filtered_sent2=[]
for w in tokenized_word2:
    if w not in stop_words2:
        filtered_sent.append(w)

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()
stemmed_words2=[]
for w in filtered_sent2:
    stemmed_words2.append(ps.stem(w))
words2 = [word for word in stemmed_words2 if word.isalpha()]
x3=[]
for i in range (0,200*150,150):
     x3.append(words1[i:i+150])
c=[2]*200



###########Merge three books and three labels
Merge=x1+x2+x3
import pandas as pd
Documents=np.asarray(Merge)
Label=a+b+c
Label=np.asarray(Label)
doc=np.c_[Documents,Label]

#######shuffling the document merged
import random
np.random.shuffle(doc)

#########separating labels and text after shuffling
AfterLabel=doc[:,-1:]
AfterDoc=doc[:,:150]

########converting AfterLabel and AfterDoc to DataFrames
df=pd.DataFrame(AfterDoc)
y=pd.DataFrame(AfterLabel)
df=[" ".join(c) for c in df.values]
df=pd.DataFrame(df)

#importing various modules of scikit-learn's
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from  sklearn.metrics  import accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report


########BOW on the text
count_vect = CountVectorizer()
X_train_counts = (count_vect.fit_transform(df[0]))

############splitting into text train and K-nearest using BOW result
X_train, X_test, y_train, y_test = train_test_split(X_train_counts ,y,test_size=.20,shuffle='true', random_state=10)
classifier = KNeighborsClassifier(n_neighbors=5)  
c1=classifier.fit(X_train , y_train)
predic=c1.predict(X_test)
r1=accuracy_score(y_test,predic)*100
print("Accuracy of BOW kNeighbour classifier",accuracy_score(y_test,predic))
confusion1=confusion_matrix(y_test, predic)
print('Confusion matrix BOW kNeighbour', confusion1)
print(classification_report(y_test, predic))
#predicted=clf.predict(bagtest)
#print(predicted)
#print(accuracy_score(y_test,predicted))

######decision tree using BOW
from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(X_train, y_train) 
y_pred1 = classifier.predict(X_test) 
r2=accuracy_score(y_test,y_pred1)*100
print("Accuracy of BOW Decision tree classifier",accuracy_score(y_test,y_pred1))
confusion2=confusion_matrix(y_test, y_pred1)
print('Confusion matrix BOW Decision tree', confusion2)
print(classification_report(y_test, y_pred1))

######SVM BOW
sv1= svm.SVC(kernel='linear', C=10)
sv1.fit(X_train, y_train)  
y_sv1 = sv1.predict(X_test)
r3=accuracy_score(y_test,y_sv1)*100
print("Accuracy of BOW SVM classifier", accuracy_score(y_test,y_sv1))
confusion3=confusion_matrix(y_test, y_sv1)
print('Confusion matrix BOW SVM', confusion3)
print(classification_report(y_test, y_sv1))

#########BOW 10 Fold
#K Fold Cross Validation
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
num_instances = len(X_train_counts.toarray())
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = LogisticRegression()
results = model_selection.cross_val_score(model, X_train_counts, y, cv=kfold)
print("Accuracy of 10-Fold cross validation BOW:" , (results.mean()*100.0, results.std()*100.0))

#####TFIDF on text
tfidf_transformer = TfidfVectorizer()
X_train_vec = tfidf_transformer.fit_transform(df[0])
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train_vec ,y,test_size=.20,shuffle='true', random_state=10 )

####k-nearest on TFIDF
classifier= KNeighborsClassifier(n_neighbors=5)  
c11=classifier.fit(X_train1 , y_train1)
predic1=c11.predict(X_test1)
t1=accuracy_score(y_test1,predic1)*100
print("Accuracy of TFIDF kNeighbour classifier",accuracy_score(y_test1,predic1))
confusion4=confusion_matrix(y_test1, predic1)
print('Confusion matrix TFIDF kNeighbour', confusion4)
print(classification_report(y_test1, predic1))

######decision tree on TFIDF
from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(X_train1, y_train1) 
y_pred2 = classifier.predict(X_test1) 
t2=accuracy_score(y_test1,y_pred2)*100
print("Accuracy of TFIDF Decision tree classifier",accuracy_score(y_test1,y_pred2))
confusion5=confusion_matrix(y_test1,y_pred2)
print('Confusion matrix TFIDF Decision tree', confusion5)
print(classification_report(y_test1,y_pred2))

###SVM Code TFIDF
sv= svm.SVC(kernel='linear', C=10)
sv.fit(X_train1, y_train1)  
y_sv = sv.predict(X_test1)
t3=accuracy_score(y_test1,y_sv)*100
print("Accuracy of TFIDF SVM classifier ", accuracy_score(y_test1,y_sv))
confusion6=confusion_matrix(y_test1, y_sv)
print('Confusion matrix TFIDF SVM', confusion6)
print(classification_report(y_test1, y_sv))

#########BOW 10 Fold
#K Fold Cross Validation
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
num_instances = len(X_train_vec.toarray())
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = LogisticRegression()
results = model_selection.cross_val_score(model, X_train_vec, y, cv=kfold)
print("Accuracy of 10-Fold cross validation TFIDF:" , (results.mean()*100.0, results.std()*100.0))


############## plotting results of accuracy of BOW
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects = ('KNeighbour', 'DecisionTree','SVM')
y_pos = np.arange(len(objects))
performance = [r1,r2,r3]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy of Classifiers in Percentage')
plt.title('BOW Classifiers Accuracy Results')
plt.show()

#########plotting results of accuarcy of TFIDF
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects1 = ('KNeighbour', 'DecisionTree','SVM')
y_pos1 = np.arange(len(objects))
performance1 = [t1,t2,t3]
plt.bar(y_pos1, performance1, align='center', alpha=0.5)
plt.xticks(y_pos1, objects1)
plt.ylabel('Accuracy of Classifiers in Percentage')
plt.title('TFIDF Classifiers Accuracy Results')
plt.show()


########Plotting confusion matrix  Error Analysis on TFIDF decision tree
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
       
df_cm = pd.DataFrame(confusion5, range(3),
                  range(3))
#plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size