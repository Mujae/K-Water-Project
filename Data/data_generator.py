# data_generator.py
import time
from argparse import ArgumentParser
import pandas as pd
import psycopg2
import numpy as np

def get_data():
    df = pd.read_csv('Database/wdata_train.csv')
    df['logTime'] = pd.to_datetime(df['logTime'], errors='coerce')
    df['year'] = df['logTime'].dt.year # 연도
    df['month'] = df['logTime'].dt.month # 월
    df.set_index('logTime', inplace=True)
    # Average over the previous 24 hours
    df['turbidity_avg24'] = df['turbidity'].rolling('24H', min_periods=1).mean()
    df.reset_index(inplace=True)
    # Separating dependent and independent variables
    df['turbidity_4H'] = df['turbidity'].shift(4)
    df['log원수탁도'] = np.log1p(df['turbidity'])
    df['t_diff'] = round(df['log원수탁도'].diff(),8)
    df.drop('log원수탁도', axis=1, inplace=True)
    df['ppm_shift_1H'] = df['target'].shift(-1)
    df.fillna(0,inplace=True)
    df['logTime'] = df['logTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df


def create_table(db_connect):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS wdata (
        id SERIAL PRIMARY KEY,
        timestamp timestamp,
        logTime timestamp,
        turbidity float8,
        alkalinity float8,
        conductivity float8,
        pH float8,
        temp float8,
        rainfall float8,
        year int,
        month int,
        turbidity_avg24 float8,
        turbidity_4H float8,
        t_diff float8,
        target float8
    );"""
    print(create_table_query)
    with db_connect.cursor() as cur:
        cur.execute(create_table_query)
        db_connect.commit()



def insert_data(db_connect, data):
    insert_row_query = f"""
    INSERT INTO wdata
        (timestamp, logTime, turbidity, alkalinity, conductivity, pH, temp, rainfall, year, month, turbidity_avg24, turbidity_4H, t_diff, target)
        VALUES (
            NOW(),
            '{data.logTime}',
            {data.turbidity},
            {data.alkalinity},
            {data.conductivity},
            {data.pH},
            {data.temp},
            {data.rainfall},
            {data.year},
            {data.month},
            {data.turbidity_avg24},
            {data.turbidity_4H},
            {data.t_diff},
            {data.target}
        );"""
    print(insert_row_query)
    with db_connect.cursor() as cur:
        cur.execute(insert_row_query)
        db_connect.commit()


def generate_data(db_connect, df):
    while True:
        insert_data(db_connect, df.sample(1).squeeze())
        time.sleep(1)


if __name__ == "__main__":
    parser = ArgumentParser()   
    parser.add_argument("--db-host", dest="db_host", type=str, default="localhost")
    args = parser.parse_args()

    db_connect = psycopg2.connect(
        user="myuser",
        password="mypassword",
        host=args.db_host,
        port=5432,
        database="mydatabase",
    )
    create_table(db_connect)
    df = get_data()
    generate_data(db_connect, df)
