FROM python:3.8-slim-buster

RUN mkdir /app0

WORKDIR /app0

ENV FLASK_APP=api

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY api.py /app0

# COPY ./model/text_clf.pickle ./model/text_clf.pickle 

ENTRYPOINT ["flask","run","--port","5006"]