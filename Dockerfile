FROM jupyter/scipy-notebook:latest
LABEL authors="mp74207"

RUN pip install -r requirements.txt
ENTRYPOINT ["top", "-b"]