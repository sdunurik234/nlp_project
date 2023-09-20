from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import telebot
import joblib
import nltk
import re
import string
from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
# nltk.download('wordnet')
# nltk.download('omw-1.4')
st = nltk.PorterStemmer()
lm = nltk.WordNetLemmatizer()

stop_words = set(stopwords.words("russian"))

def tazartu(text):
    n_zhok = text.replace("\n", "")
    text_tazar = "".join([i for i in n_zhok if i not in string.punctuation])
    token = re.split("\W+", text_tazar)
    stop_w = [j.lower() for j in token if j not in stop_words]
    sting = [st.stem(w) for w in stop_w]
    lmize = " ".join([lm.lemmatize(k) for k in sting])
    return lmize

svm = joblib.load("model_svm.joblib")

bot = telebot.TeleBot("6617659535:AAFR8Km7as8yftAdOSGlCa8R33i1hYsQKEU")

@bot.message_handler(commands=["start"])
def index(message):
    bot.send_message(message.chat.id, "Напиши что нибудь чтобы определить negative или positive")

@bot.message_handler()
def mes(message):
    if message.text:
        text = tazartu(message.text)
        res = svm.predict([text])
        if sum(res) == 1:
            bot.send_message(message.chat.id, "Қотақбас пиздец шығарсын сен")
        else:
            bot.send_message(message.chat.id, "Ақи брат мен сені сүйем")


bot.polling(none_stop=True)