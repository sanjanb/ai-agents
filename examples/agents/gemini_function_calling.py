"""
Function calling with the Gemini API over a local SQLite database.

Prereqs:
  - pip install -r examples/agents/requirements.txt
  - Set environment variable GOOGLE_API_KEY

Run:
  python examples/agents/gemini_function_calling.py
"""

import os
import sqlite3
from typing import Any, Dict, List

try:
    from google import genai
    from google.genai import types
except Exception as e:  # pragma: no cover
    raise SystemExit("google-genai is required. Install requirements first.")


def init_db(path: str = "sample.db") -> sqlite3.Connection:
    if os.path.exists(path):
        os.remove(path)
    db = sqlite3.connect(path)
    db.executescript(
        """
        CREATE TABLE products (product_id INTEGER PRIMARY KEY, product_name TEXT, price REAL);
        CREATE TABLE staff (staff_id INTEGER PRIMARY KEY, first_name TEXT, last_name TEXT);
        CREATE TABLE orders (
          order_id INTEGER PRIMARY KEY, customer_name TEXT, staff_id INT, product_id INT,
          FOREIGN KEY(staff_id) REFERENCES staff(staff_id),
          FOREIGN KEY(product_id) REFERENCES products(product_id)
        );
        INSERT INTO products(product_name, price) VALUES ('Laptop',799.99),('Keyboard',129.99),('Mouse',29.99);
        INSERT INTO staff(first_name,last_name) VALUES ('Alice','Smith'),('Bob','Johnson');
        INSERT INTO orders(customer_name,staff_id,product_id) VALUES ('David Lee',1,1),('Emily Chen',2,2);
        """
    )
    return db


def list_tables(db: sqlite3.Connection) -> List[str]:
    cur = db.cursor()
    rows = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [r[0] for r in rows]


def describe_table(db: sqlite3.Connection, table: str) -> List[Dict[str, Any]]:
    cur = db.cursor()
    rows = cur.execute(f"PRAGMA table_info({table})").fetchall()
    return [{"cid": r[0], "name": r[1], "type": r[2], "notnull": r[3]} for r in rows]


def run_sql(db: sqlite3.Connection, query: str) -> List[Dict[str, Any]]:
    assert query.strip().lower().startswith("select"), "Only SELECT allowed"
    cur = db.cursor()
    rows = cur.execute(query).fetchall()
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in rows]


def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GOOGLE_API_KEY in your environment before running.")

    db = init_db()

    client = genai.Client(api_key=api_key)
    model = client.models.get(name="gemini-1.5-flash")

    # Bind typed function schemas
    tools = [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="list_tables",
                    description="List available SQLite tables.",
                    parameters=types.Schema(type=types.Type.OBJECT, properties={}),
                ),
                types.FunctionDeclaration(
                    name="describe_table",
                    description="Describe columns for a given table.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={"table": types.Schema(type=types.Type.STRING)},
                        required=["table"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="run_sql",
                    description="Execute a read-only SQL query and return rows as dicts.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={"query": types.Schema(type=types.Type.STRING)},
                        required=["query"],
                    ),
                ),
            ]
        )
    ]

    user_query = (
        "Show me product names and prices from the products table ordered by price desc"
    )

    resp = client.models.generate_content(
        model=model.name, contents=[user_query], config=types.GenerateContentConfig(tools=tools)
    )

    while True:
        calls = resp.function_calls or []
        if not calls:
            print("Model response:\n", resp.text)
            break

        call = calls[0]
        fn_name = call.name
        args = dict(call.args) if call.args else {}

        if fn_name == "list_tables":
            result = list_tables(db)
        elif fn_name == "describe_table":
            result = describe_table(db, **args)
        elif fn_name == "run_sql":
            result = run_sql(db, **args)
        else:
            result = {"error": f"unknown function {fn_name}"}

        resp = client.models.generate_content(
            model=model.name,
            contents=[
                types.Content(role="user", parts=[types.Part(text=user_query)]),
                types.Content(
                    role="tool",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(name=fn_name, response=result)
                        )
                    ],
                ),
            ],
            config=types.GenerateContentConfig(tools=tools),
        )


if __name__ == "__main__":
    main()
