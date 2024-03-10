# data_subscriber.py
from json import loads
import pandas as pd
import psycopg2
import requests
from kafka import KafkaConsumer

def create_table(db_connect):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS water_prediction(
        id SERIAL PRIMARY KEY,
        timestamp timestamp,
        target float,
        turbidity_1h float,
        turbidity_2h float,
        turbidity_3h float,
        turbidity_4h float
    );"""
    print(create_table_query)
    with db_connect.cursor() as cur:
        cur.execute(create_table_query)
        db_connect.commit()


def insert_data(db_connect, data):
    insert_row_query = f"""
    INSERT INTO water_prediction
        (timestamp, target, turbidity_1h, turbidity_2h, turbidity_3h, turbidity_4h)
        VALUES (
            '{data["timestamp"]}',
            {data["target"]},
            {data["turbidity_1h"]},
            {data["turbidity_2h"]},
            {data["turbidity_3h"]},
            {data["turbidity_4h"]}
        );"""
    print(insert_row_query)
    with db_connect.cursor() as cur:
        cur.execute(insert_row_query)
        db_connect.commit()


def subscribe_data(db_connect, consumer):
    for msg in consumer:
        print(
            f"Topic : {msg.topic}\n"
            f"Partition : {msg.partition}\n"
            f"Offset : {msg.offset}\n"
            f"Key : {msg.key}\n"
            f"Value : {msg.value}\n",
        )

        msg.value["payload"].pop("id")
        msg.value["payload"].pop("target")
        ts = msg.value["payload"].pop("timestamp")
        dic1 = msg.value["payload"].copy()
        dic2 = msg.value["payload"].copy()
        dic1.pop("turbidity_4h")
        dic1["pH"] =dic1["ph"]
        dic1.pop("ph")
        dic1.pop("t_diff")
        dic1.pop("logtime")
        dic1.pop("rainfall")
        dic2.pop("logtime")
        dic2.pop("turbidity_avg24")
        dic2.pop("conductivity")
        dic2.pop("ph")
        dic2.pop("alkalinity")
        dic2.pop("month")
        dic2.pop("year")
        response = requests.post(
            url="http://api-serving:8000/predict_ppm",
            json=dic1,
            headers={"Content-Type": "application/json"},
        ).json()

        response_tur1 = requests.post(
            url="http://api-serving:8000/predict_turbidity_1H",
            json=dic2,
            headers={"Content-Type": "application/json"},
        ).json()

        response_tur2 = requests.post(
            url="http://api-serving:8000/predict_turbidity_2H",
            json=dic2,
            headers={"Content-Type": "application/json"},
        ).json()

        response_tur3 = requests.post(
            url="http://api-serving:8000/predict_turbidity_3H",
            json=dic2,
            headers={"Content-Type": "application/json"},
        ).json()

        response_tur4 = requests.post(
            url="http://api-serving:8000/predict_turbidity_4H",
            json=dic2,
            headers={"Content-Type": "application/json"},
        ).json()

        response7 = dict()
        print(response)
        response7["target"] = response["target"]
        response7["timestamp"] = ts
        response7["turbidity_1h"] = response_tur1["target"]
        response7["turbidity_2h"] = response_tur2["target"]
        response7["turbidity_3h"] = response_tur3["target"]
        response7["turbidity_4h"] = response_tur4["target"]
        print(response7)
        insert_data(db_connect, response7)


if __name__ == "__main__":
    db_connect = psycopg2.connect(
        user="targetuser",
        password="targetpassword",
        host="target-postgres-server",
        port=5432,
        database="targetdatabase",
    )
    create_table(db_connect)

    consumer = KafkaConsumer(
        "postgres-source-wdata",
        bootstrap_servers="broker:29092",
        auto_offset_reset="earliest",
        group_id="wconsumer-group",
        value_deserializer=lambda x: loads(x),
    )
    subscribe_data(db_connect, consumer)