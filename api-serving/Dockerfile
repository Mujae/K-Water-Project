FROM python:3.9-slim

WORKDIR /usr/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1


RUN pip install -U pip &&\
    pip install mlflow==1.30.0 pandas scikit-learn "fastapi[all]" psycopg2-binary tensorflow lightgbm numpy

COPY schemas.py schemas.py
COPY app.py app.py
COPY LGBM_15down/ LGBM_15down/
COPY LGBM_15up_1/ LGBM_15up_1/
COPY LGBM_15up_2/ LGBM_15up_2/
COPY LGBM_15up_3/ LGBM_15up_3/
COPY LGBM_15up_4/ LGBM_15up_4/
COPY LGBM_15up_5/ LGBM_15up_5/
COPY LGBM_15up_6/ LGBM_15up_6/
COPY LSTM1/ LSTM1/
COPY LSTM2/ LSTM2/
COPY LSTM3/ LSTM3/
COPY LSTM4/ LSTM4/
COPY LSTM_scaler.joblib /usr/app/LSTM_scaler.joblib
COPY spk.joblib /usr/app/spk.joblib


CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--reload"]