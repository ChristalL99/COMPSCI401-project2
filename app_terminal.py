import pandas as pd
import urllib.request
from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from github import Github

# References: https://www.w3cschool.cn/flask/flask_http_methods.html
# Discussed the idea with Zhengge Tang

# Creation of the Flask app
app = Flask(__name__)

g = Github()
repo = g.get_repo("ChristalL99/COMPSCI401-project2")
model_commits = repo.get_commits(path = './text_clf.pickle')
code_commits = repo.get_commits(path = './app_terminal.py')

version = str(code_commits.totalCount)
model_date = str(pd.to_datetime(model_commits[0].commit.committer.date))

# Load the pipeline
model_url = 'https://github.com/ChristalL99/COMPSCI401-project2/blob/main/text_clf.pickle?raw=True'
clf_filename, headers = urllib.request.urlretrieve(model_url, filename = "text_clf.pickle")


@app.route('/api/american', methods=['POST'])
def predict():
    content = request.get_json(force=True)
    text = [content['text']]
    result = app.model.predict(text)[0]
    return jsonify(is_american = str(result), version = version, model_date = model_date)

if __name__ == '__main__':
    app.run(port=5006, debug=True)



# wget --server-response --output-document response.out --header='Content-Type: application/json' --post-data '{"text": "#covid19 new york"}' http://localhost:5006/api/american
