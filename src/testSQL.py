import sqlite3
import pandas as pd


def run_diagnostic_query(db_path: str, query: str) -> pd.DataFrame:
    """
    Connects to the SQLite database, runs the provided SQL query, and returns the results as a DataFrame.

    Args:
        db_path (str): The path to the SQLite database file.
        query (str): The SQL query to execute.

    Returns:
        pd.DataFrame: The results of the query as a DataFrame.
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)

    try:
        # Execute the query and fetch results into a DataFrame
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"Error executing query: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error
    finally:
        # Ensure the connection is closed even if an error occurs
        conn.close()
