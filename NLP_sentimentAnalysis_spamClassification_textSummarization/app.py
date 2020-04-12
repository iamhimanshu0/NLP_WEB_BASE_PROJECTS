from flask import Flask, render_template, request,url_for

# nlp library
import spacy
from sklearn.externals import joblib
from spacy.lang.en.stop_words import STOP_WORDS
import string

# summarize
from gensim.summarization import summarize

# load the spacy english model
nlp = spacy.load('en')
punct = string.punctuation
stopwords = list(STOP_WORDS)

# model load sentiment analysis:
sen_model = joblib.load('models/sentiment.pkl')

# model load spam classification
spam_model = joblib.load('models/spam.pkl')

# news classifier
n_clf = joblib.load('models/news_classifier.pkl')

app = Flask(__name__)


# home page
@app.route('/')
def index():
	return render_template('home.html')


# sentiment analysis
@app.route('/nlpsentiment')
def sentiment_nlp():
	return render_template('sentiment.html')


@app.route('/sentiment',methods = ['POST','GET'])
def sentiment():
	if request.method == 'POST':
		message = request.form['message']
		# Machine learning analysiser
		pred = sen_model.predict([message])
		return render_template('sentiment.html', prediction=pred)


# spam
@app.route('/nlpspam')
def spam_nlp():
	return render_template('spam.html')

# spam classification
@app.route('/spam',methods= ['POST','GET'])
def spam():
	if request.method == 'POST':
		message = request.form['message']
		pred = spam_model.predict([message])
		return render_template('spam.html',prediction=pred)


# summarize
@app.route('/nlpsummarize')
def summarize_nlp():
	return render_template('summarize.html')

# spam classification
@app.route('/summarize',methods= ['POST','GET'])
def sum_route():
	if request.method == 'POST':
		message = request.form['message']
		sum_message = summarize(message)
		return render_template('summarize.html',original = message, prediction=sum_message)


@app.route('/newsclf')
def news_classifier():
	return render_template('news.html')

@app.route('/newsclassifier',methods=['POST','GET'])
def news_clf():
	if request.method == 'POST':
		message = request.form['message']
		pred = n_clf.predict([message])
		return render_template('news.html',prediction=pred)


if __name__ == '__main__':
	app.run(debug=True)