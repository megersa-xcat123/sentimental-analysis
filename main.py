#import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score,accuracy_score,recall_score, precision_score
from tensorflow_core.python.keras.layers import SpatialDropout1D
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import pickle
import package as pkg

##read the csv file using pandas framework and download

df_train = pd.read_csv('Amh-Dataset/Training-dataset - Sheet1.csv')
df_test = pd.read_csv('Amh-Dataset/Test-dataset - Sheet1(1).csv')
# output the read dataset
print((df_train.head))
#Data pre-processing and Cleansing

df_train = pkg.clean_df(df_train)
df_test = pkg.clean_df(df_test)
#print("cleaned dataset")
#print(df_train[5:])


#df_test = clean_df(df_test)


clear_config = {
    'remove_url': True,
    'remove_mentions': True,
    'lowercase': True,
    'demojify': True
}

df_train['text'] = df_train['text'].apply(pkg.clean_text, args=(clear_config,))
df_test['text'] = df_test['text'].apply(pkg.clean_text, args=(clear_config,))

# character level normalization
#method to normalize a character level mismatch such as ጸሀይ and ፀሐይ

df_train['text'] = df_train['text'].apply(lambda x: pkg.normalize_char_level_missmatch(x))
#df_test['text'] = df_test['text'].apply(lambda x: normalize_char_level_missmatch(x))


#MODEL 1: USING NAIVE BAYES MODEL AND COUNT VECTORIZER

#assigne the train data independent data

X_train = df_train['text'].values
#dependent data
y_train = df_train['sentiment'].values

X_test, y_test = df_test['text'].values, df_test['sentiment'].values

matrix = CountVectorizer(analyzer='word', max_features=1000, ngram_range=(1, 3), lowercase=True)
#print(X_train.shape)
#print("Vocabulary...")
vector = matrix.fit(X_train)
#print(print("Vocabulary: ", vector.vocabulary_))
#p


#TfIDF
from  sklearn.feature_extraction.text import TfidfVectorizer
##
vectorizer = TfidfVectorizer()
vectorizer.fit(X_train)
X = vectorizer.transform(X_train)


X_train = matrix.fit_transform(X_train).toarray() #scaling
X_test = matrix.fit_transform(X_test).toarray()
np.set_printoptions(threshold=sys.maxsize)

#document = ["የታመነ ሰው እጅግ ይባረካል"]

#for i in range(2):
#    print(X_train[i])

#initialize and train the model
print("*****Gaussian  Algorithm*****")
classifier = GaussianNB()
classifier.fit(X_train,y_train)
#predict Class
print("pickle model 12")
y_pred = classifier.predict(X_test)
#print(y_pred)
#measure accuracy
#accuracy
acc = accuracy_score(y_test,y_pred)
print("the accuracy value is: {}%".format(100*acc))
#make a pickle file for model
print("pickle model dump")
pickle.dump(classifier, open("model.pkl","wb"))
#classification report and confusion matrix

print(classification_report(y_test,y_pred, target_names=['Positive','Negative','Neutral']))

#SVC
print("*****SVC Algorithm*****")
from sklearn import svm
clf_svm = svm.SVC()
clf_svm.fit(X_train,y_train)
prediction = clf_svm.predict(X_test)
#print(prediction)
#accuracy
acc = accuracy_score(y_test,prediction)
print("the accuracy value is: {}%".format(100*acc))

#classification report and confusion matrix
print(classification_report(y_test,prediction, target_names=['Positive','Negative','Neutral']))


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
trainX, testX, trainY,testY = train_test_split(X, y_train,test_size=0.2, stratify=y_train, random_state=2)
#print("TFIDF value", X)



logic_clf = LogisticRegression()
logic_clf.fit(trainX,trainY)
X_train_prediction = logic_clf.predict(trainX)
training_data_accuracy = accuracy_score(X_train_prediction, trainY)
print("Accuracy score ", training_data_accuracy)


print("Testing Data accuracy")
X_test_prediction = logic_clf.predict(testX)
testing_data_accuracy = accuracy_score(X_test_prediction, testY)
print("Accuracy score test ", testing_data_accuracy)


data = testX[1]
x_pred = logic_clf.predict(data)
if (x_pred[0] == 'Positive'):
    print("The data is: positive")
elif (x_pred[0]=='Negative'):
    print("The data is: Negative")
else:
    print("The data is: Neutral")

print(testY[1])

