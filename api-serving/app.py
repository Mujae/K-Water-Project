# app.py
import os
import mlflow
import pandas as pd
from fastapi import FastAPI
from schemas import PredictIn_ppm, PredictOut_ppm, PredictIn_tur, PredictOut_tur
import joblib
import numpy as np
import psycopg2

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "myuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mypassword")
DB_NAME = os.getenv("DB_NAME", "mydatabase")

db_connect = psycopg2.connect(
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=5432,
    database=DB_NAME,
)

def get_lgbm_15under():
    model1 = mlflow.sklearn.load_model(model_uri="./LGBM_15down")
    return model1

def get_lgbm1_15up():
    model2 = mlflow.sklearn.load_model(model_uri="./LGBM_15up_1")
    return model2

def get_lgbm2_15up():
    model3 = mlflow.sklearn.load_model(model_uri="./LGBM_15up_2")
    return model3

def get_lgbm3_15up():
    model4 = mlflow.sklearn.load_model(model_uri="./LGBM_15up_3")
    return model4

def get_lgbm4_15up():
    model5 = mlflow.sklearn.load_model(model_uri="./LGBM_15up_4")
    return model5

def get_lgbm5_15up():
    model6 = mlflow.sklearn.load_model(model_uri="./LGBM_15up_5")
    return model6

def get_lgbm6_15up():
    model7 = mlflow.sklearn.load_model(model_uri="./LGBM_15up_6")
    return model7

def get_LSTM1():
    model8 = mlflow.sklearn.load_model(model_uri="./LSTM1")
    return model8
def get_LSTM2():
    model9 = mlflow.sklearn.load_model(model_uri="./LSTM2")
    return model9
def get_LSTM3():
    model10 = mlflow.sklearn.load_model(model_uri="./LSTM3")
    return model10
def get_LSTM4():
    model11 = mlflow.sklearn.load_model(model_uri="./LSTM4")
    return model11

model_15d = get_lgbm_15under()
model_lgbm1 =  get_lgbm1_15up()
model_lgbm2 =  get_lgbm2_15up()
model_lgbm3 =  get_lgbm3_15up()
model_lgbm4 =  get_lgbm4_15up()
model_lgbm5 =  get_lgbm5_15up()
model_lgbm6 =  get_lgbm6_15up()
model_LSTM1 = get_LSTM1()
model_LSTM2 = get_LSTM2()
model_LSTM3 = get_LSTM3()
model_LSTM4 = get_LSTM4()


# Create a FastAPI instance
app = FastAPI()


@app.post("/predict_ppm", response_model=PredictOut_ppm)
def predict(data: PredictIn_ppm) -> PredictOut_ppm:
    df = pd.DataFrame([data.dict()])
    # Turbidity>15 -> We use cluster (k=6). It improves the performance.
    if df['turbidity'].iloc[0] >= 15:
        original_data = df[['turbidity', 'alkalinity', 'conductivity', 'pH', 'temp']].copy()
        pipeline = joblib.load('/usr/app/spk.joblib')
        cluster = pipeline.predict(original_data)
        if cluster == 0:
            pred = model_lgbm1.predict(df).item()
        elif cluster == 1:
            pred = model_lgbm2.predict(df).item()
        elif cluster == 2:
            pred = model_lgbm3.predict(df).item()
        elif cluster == 3:
            pred = model_lgbm4.predict(df).item()
        elif cluster == 4:
            pred = model_lgbm5.predict(df).item()
        else:
            pred = model_lgbm6.predict(df).item()
    else:  
        pred = model_15d.predict(df).item()
    return PredictOut_ppm(target=pred)

seq_len = 48   # sequence length = past days for future prediction.
scaler = joblib.load('/usr/app/LSTM_scaler.joblib') # scaler we made before

# In performance One turbidity model < 4 turbidity model
@app.post("/predict_turbidity_1H", response_model=PredictOut_tur)
def predict(data: PredictIn_tur) -> PredictOut_tur:
    df = pd.DataFrame([data.dict()])
    query = "SELECT * FROM wdata ORDER BY timestamp DESC LIMIT 47"
    df2 = pd.read_sql(query, db_connect)
    df2 = df2[['turbidity', 'temp', 'rainfall','turbidity_4h', 't_diff']]
    result_df = pd.concat([df2, df], ignore_index=True)
    result_df = scaler.transform(result_df)
    result_df = np.array(result_df).reshape(1, 48, -1)
    pred = model_LSTM1.predict(result_df).item()*scaler.scale_[0] + scaler.mean_[0]
    print(pred)

    return PredictOut_tur(target=pred)

@app.post("/predict_turbidity_2H", response_model=PredictOut_tur)
def predict(data: PredictIn_tur) -> PredictOut_tur:
    df = pd.DataFrame([data.dict()])
    query = "SELECT * FROM wdata ORDER BY timestamp DESC LIMIT 47"
    df2 = pd.read_sql(query, db_connect)
    df2 = df2[['turbidity', 'temp', 'rainfall','turbidity_4h', 't_diff']]
    result_df = pd.concat([df2, df], ignore_index=True)
    result_df = scaler.transform(result_df)
    result_df = np.array(result_df).reshape(1, 48, -1)
    pred = model_LSTM2.predict(result_df).item()*scaler.scale_[0] + scaler.mean_[0]
    print(pred)
    return PredictOut_tur(target=pred)

@app.post("/predict_turbidity_3H", response_model=PredictOut_tur)
def predict(data: PredictIn_tur) -> PredictOut_tur:
    df = pd.DataFrame([data.dict()])
    query = "SELECT * FROM wdata ORDER BY timestamp DESC LIMIT 47"
    df2 = pd.read_sql(query, db_connect)
    df2 = df2[['turbidity', 'temp', 'rainfall','turbidity_4h', 't_diff']]
    result_df = pd.concat([df2, df], ignore_index=True)
    result_df = scaler.transform(result_df)
    result_df = np.array(result_df).reshape(1, 48, -1)
    pred = model_LSTM3.predict(result_df).item()*scaler.scale_[0] + scaler.mean_[0]
    print(pred)
    return PredictOut_tur(target=pred)

@app.post("/predict_turbidity_4H", response_model=PredictOut_tur)
def predict(data: PredictIn_tur) -> PredictOut_tur:
    df = pd.DataFrame([data.dict()])
    query = "SELECT * FROM wdata ORDER BY timestamp DESC LIMIT 47"
    df2 = pd.read_sql(query, db_connect)
    df2 = df2[['turbidity', 'temp', 'rainfall','turbidity_4h', 't_diff']]
    result_df = pd.concat([df2, df], ignore_index=True)
    result_df = scaler.transform(result_df)
    result_df = np.array(result_df).reshape(1, 48, -1)
    pred = model_LSTM4.predict(result_df).item()*scaler.scale_[0] + scaler.mean_[0]
    print(pred)
    return PredictOut_tur(target=pred)
