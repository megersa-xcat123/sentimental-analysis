#import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score,accuracy_score,recall_score, precision_score
from tensorflow_core.python.keras.layers import SpatialDropout1D
from  sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  LSTM, Dense,Dropout, Embedding
import sys
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import re
import pickle
import tweepy
from tweepy import  OAuthHandler

#CONSTANT VARIABLE
embedding_vector_length = 32


def clean_df(df):
    """ removes null values and resets index"""
    #remove null values
    df = df.dropna()
    #drop repeated rows and (drop rows with similar text)
    df = df.drop_duplicates(subset='text', keep="first")
    df = df.reset_index(drop=True)
    return  df
def clean_text (row, options):
    """ Removes url, mentions,emoji and uppercase from amharic-text"""
    if options['lowercase']:
        row = row.lower()
    if options['remove_url']:
        row = re.sub(r"(?:\@https?\://)\S+","",row)
    if options['remove_mentions']:
        row = re.sub("@[A-Za-z0-9_]+", "", row)
    if options['demojify']:
        emoj = re.compile("["
                          u"\U0001F600-\U0001F64F"  # emoticons
                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                          u"\U00002500-\U00002BEF"  # chinese char
                          u"\U00002702-\U000027B0"
                          u"\U00002702-\U000027B0"
                          u"\U000024C2-\U0001F251"
                          u"\U0001f926-\U0001f937"
                          u"\U00010000-\U0010ffff"
                          u"\u2640-\u2642"
                          u"\u2600-\u2B55"
                          u"\u200d"
                          u"\u23cf"
                          u"\u23e9"
                          u"\u231a"
                          u"\ufe0f"  # dingbats
                          u"\u3030"
                          "]+", re.UNICODE)
        row = re.sub(emoj, '', row)
    return  row
#method to normalize a character level mismatch such as ጸሀይ and ፀሐይ

def normalize_char_level_missmatch(input_token):
    rep1=re.sub('[ሃኅኃሐሓኻ]','ሀ',input_token)
    rep2=re.sub('[ሑኁዅ]','ሁ',rep1)
    rep3=re.sub('[ኂሒኺ]','ሂ',rep2)
    rep4=re.sub('[ኌሔዄ]','ሄ',rep3)
    rep5=re.sub('[ሕኅ]','ህ',rep4)
    rep6=re.sub('[ኆሖኾ]','ሆ',rep5)
    rep7=re.sub('[ሠ]','ሰ',rep6)
    rep8=re.sub('[ሡ]','ሱ',rep7)
    rep9=re.sub('[ሢ]','ሲ',rep8)
    rep10=re.sub('[ሣ]','ሳ',rep9)
    rep11=re.sub('[ሤ]','ሴ',rep10)
    rep12=re.sub('[ሥ]','ስ',rep11)
    rep13=re.sub('[ሦ]','ሶ',rep12)
    rep14=re.sub('[ዓኣዐ]','አ',rep13)
    rep15=re.sub('[ዑ]','ኡ',rep14)
    rep16=re.sub('[ዒ]','ኢ',rep15)
    rep17=re.sub('[ዔ]','ኤ',rep16)
    rep18=re.sub('[ዕ]','እ',rep17)
    rep19=re.sub('[ዖ]','ኦ',rep18)
    rep20=re.sub('[ጸ]','ፀ',rep19)
    rep21=re.sub('[ጹ]','ፁ',rep20)
    rep22=re.sub('[ጺ]','ፂ',rep21)
    rep23=re.sub('[ጻ]','ፃ',rep22)
    rep24=re.sub('[ጼ]','ፄ',rep23)
    rep25=re.sub('[ጽ]','ፅ',rep24)
    rep26=re.sub('[ጾ]','ፆ',rep25)
    #Normalizing words with Labialized Amharic characters such as በልቱዋል or  በልቱአል to  በልቷል
    rep27=re.sub('(ሉ[ዋአ])','ሏ',rep26)
    rep28=re.sub('(ሙ[ዋአ])','ሟ',rep27)
    rep29=re.sub('(ቱ[ዋአ])','ቷ',rep28)
    rep30=re.sub('(ሩ[ዋአ])','ሯ',rep29)
    rep31=re.sub('(ሱ[ዋአ])','ሷ',rep30)
    rep32=re.sub('(ሹ[ዋአ])','ሿ',rep31)
    rep33=re.sub('(ቁ[ዋአ])','ቋ',rep32)
    rep34=re.sub('(ቡ[ዋአ])','ቧ',rep33)
    rep35=re.sub('(ቹ[ዋአ])','ቿ',rep34)
    rep36=re.sub('(ሁ[ዋአ])','ኋ',rep35)
    rep37=re.sub('(ኑ[ዋአ])','ኗ',rep36)
    rep38=re.sub('(ኙ[ዋአ])','ኟ',rep37)
    rep39=re.sub('(ኩ[ዋአ])','ኳ',rep38)
    rep40=re.sub('(ዙ[ዋአ])','ዟ',rep39)
    rep41=re.sub('(ጉ[ዋአ])','ጓ',rep40)
    rep42=re.sub('(ደ[ዋአ])','ዷ',rep41)
    rep43=re.sub('(ጡ[ዋአ])','ጧ',rep42)
    rep44=re.sub('(ጩ[ዋአ])','ጯ',rep43)
    rep45=re.sub('(ጹ[ዋአ])','ጿ',rep44)
    rep46=re.sub('(ፉ[ዋአ])','ፏ',rep45)
    rep47=re.sub('[ቊ]','ቁ',rep46) #ቁ can be written as ቊ
    rep48=re.sub('[ኵ]','ኩ',rep47) #ኩ can be also written as ኵ
    return rep48

# model = Sequential()
# def create_model():
#     model.add(Embedding(vocab_size, embedding_vector_length,  input_length=200))
#     model.add(SpatialDropout1D(0.25))
#     model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
#     model.add(Dropout(0.2))
#     model.add(Dense(1, activation='sigmoid'))#from sigmoid to softmax
#
#     model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
#
# print(model.summary())


class Base:
    """Base class that houses common utilities for reading in test data
    and calculating model accuracy and F1 scores.
    """
    def __init__(self) -> None:
        pass

    def read_data(self, fname: str, lower_case: bool=False,
                  colnames=['truth', 'text']) -> pd.DataFrame:
        "Read in test data into a Pandas DataFrame"
        df = pd.read_csv(fname, sep='\t', header=None, names=colnames)
        df['truth'] = df['truth'].str.replace('__label__', '')
        # Categorical data type for truth labels
        df['truth'] = df['truth'].astype(int).astype('category')

        # Optional lowercase for test data (if model was trained on lowercased text)
        if lower_case:
            df['text'] = df['text'].str.lower()
        return df

    def accuracy(self, df: pd.DataFrame) -> None:
        "Prediction accuracy (percentage) and F1 score"

        acc = accuracy_score(df['sentiment'], df['pred'])*100
        f1 = f1_score(df['sentiment'], df['pred'], average='macro')*100

        recall = recall_score(df['sentiment'], df['pred'], average='macro')*100
        precision = precision_score(df['sentiment'], df['pred'], average='macro')*100
        newf1 = 2 * recall * precision / (recall + precision)

        df.to_csv("result.csv",index = False)

        print(len(df))
        print(
            "Accuracy: {:.3f}\nMacro F1-score: {:.3f}\nMacro recall: {:.3f}\nMacro precission: {:.3f}\nNew F1 MAcro: {:.3f}".format(
                acc, f1, recall, precision, newf1))
        print("{:.2f} & {:.2f} & {:.2f} & {:.2f}".format(acc, recall, precision, newf1))

class TextBlobSentiment(Base):
    """Predict sentiment scores using TextBlob.
    https://textblob.readthedocs.io/en/dev/
    """
    def __init__(self, model_file: str=None) -> None:
        super().__init__()

    def score(self, text: str) -> float:
        # pip install textblob
        from textblob import TextBlob
        return TextBlob(text).sentiment.polarity

    def predict(self, df_train, df_test, lower_case: bool) -> pd.DataFrame:
        df_test['score'] = df_test['tweet'].apply(self.score)
        # Convert float score to category based on binning
        df_test['pred'] = pd.cut(df_test['score'],
                                 bins=3,
                                 labels=["negative", "neutral", "positive"])
        df = df_test.drop('score', axis=1)
        return df