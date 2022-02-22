import pandas as pd
import urllib.request
from flask import Flask, jsonify, request, redirect, url_for, render_template
from github import Github
import pickle

# References: https://www.w3cschool.cn/flask/flask_http_methods.html
# Discussed the idea with Zhengge Tang, Chenglin Zhang

# Creation of the Flask app
app = Flask(__name__)

github = Github()
repo = github.get_repo("ChristalL99/COMPSCI401-project2")
model_commits = repo.get_commits(path = './text_clf.pickle')
code_commits = repo.get_commits(path = './app_terminal.py')

version = str(code_commits.totalCount)
model_date = str(pd.to_datetime(model_commits[0].commit.committer.date))

# Load the pipeline
model_url = 'https://github.com/ChristalL99/COMPSCI401-project2/blob/main/text_clf.pickle?raw=True'
clf_filename, headers = urllib.request.urlretrieve(model_url, filename = "text_clf.pickle")

model = pickle.load(open(clf_filename, 'rb'))

@app.route('/api/american', methods=['POST'])
def predict():
    content = request.get_json(force=True)
    text = [content['text']]
    result = model.predict(text)[0]
    return jsonify(is_american = str(result), version = version, model_date = model_date)

@app.route('/')
def index(): 
    return "Index Page"

@app.route('/success/<text>')
def success(text):
    text = [text]
    result = model.predict(text)[0]
    return jsonify(is_american = int(result), version = version, model_date = model_date)

@app.route('/api/submit', methods=['POST', 'GET'])
def input_text():
    if request.method == 'POST':
        input_text = request.form['in']
        return redirect(url_for('success', text = input_text))

    return render_template('app.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5006, debug=True)



# wget --server-response --output-document response.out --header='Content-Type: application/json' --post-data '{"text": "#covid19 new york"}' http://localhost:5006/api/american
