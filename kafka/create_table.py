# create_table.py
import psycopg2


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


if __name__ == "__main__":
    db_connect = psycopg2.connect(
        user="targetuser",
        password="targetpassword",
        host="target-postgres-server",
        port=5432,
        database="targetdatabase",
    )
    create_table(db_connect)