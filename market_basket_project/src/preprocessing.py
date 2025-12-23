import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

def load_data(filepath):
    """
    Loads the groceries dataset.
    """
    df = pd.read_csv(filepath)
    # Ensure Date is datetime just in case (though we group by it as string/object usually works too, but consistency is good)
    # The format in the CSV is DD-MM-YYYY based on the view_file output (e.g., 21-07-2015)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    return df

def get_transactions(df):
    """
    Groups data by Member_number and Date to create transactions.
    Returns a list of lists (transactions).
    """
    # Group by Member_number and Date, then list the itemDescriptions
    transactions = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).tolist()
    return transactions

def encode_transactions(transactions):
    """
    Encodes the list of transactions into a boolean (one-hot) DataFrame
    compatible with mlxtend.
    """
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    return df
