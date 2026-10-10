import pandas as pd

def drop_duplicates(data: dict) -> list:
    """
    Returns [rows_before, rows_after, cleaned_data], with integer counts and a dictionary of lists.
    """
    
    df = pd.DataFrame(data)

    rows_before = len(df)
    df = df.drop_duplicates()
    rows_after = len(df)

    return [rows_before, rows_after, df.to_dict(orient="list")]