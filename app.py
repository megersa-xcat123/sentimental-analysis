from flask import Flask, request,render_template, jsonify
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import  load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from Sentimental_Analysis_DL import sentiment_label
import  pickle
import  package as pkg
#create flask APP
from textblob import TextBlob
filename = "Saved_Model.h5"
app = Flask(__name__)
#load the model
model = pickle.load(open("model.pkl", "rb"))
loaded_model = load_model(filename)
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=['POST','GET'])
def predict():
    if request.method=='POST':
        query = request.form['amh']
        tokenizer = Tokenizer()
         #sentiment_label =
        tw = tokenizer.texts_to_sequences([query])
        tw = pad_sequences(tw,maxlen=200)
        #prediction = int(model.predict(tw).round().item())
        #acc = loaded_model.evaluate()
        prediction = int(loaded_model.predict(tw).round().item())
        value = sentiment_label[1][prediction]
        vectorizer = TfidfVectorizer()
        vectorizer.fit(query)
        X = vectorizer.transform(query)









       # blob = pkg.TextBlobSentiment()
        #result = blob.predict(query)
       # result
        #
        # sent = blob.sentiment
        #
        # pol = blob.sentiment.polarity
        #
        #
        #
        # if pol > 0:
        #     text_sentiment = "positive"
        # elif pol == 0:
        #     text_sentiment = "neutral"
        # else:
        #     text_sentiment = "negative"
        #
        # if prediction > 0:
        #     text_sentiment = "Positive"
        # elif prediction == 0:
        #     text_sentiment = "Neutral"
        # else:
        #     text_sentiment = "Negative"
    return  render_template("index.html",msg=query,prediction_test = "the text status is: {}".format(value))


if __name__ == '__main__':
    app.run(debug=False)