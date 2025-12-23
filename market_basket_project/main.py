import sys
import os
import matplotlib.pyplot as plt

# Add the current directory to path so we can import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_data, get_transactions, encode_transactions
from src.eda import plot_top_products
from src.model import get_frequent_itemsets, get_association_rules

def main():
    # 1. Load Data
    print("Loading data...")
    # Using the path relative to where we run the script (assuming running from mining folder or internal)
    # The user file is at c:\Users\user\Desktop\mining\Groceries_dataset.csv
    # We are writing this script to c:\Users\user\Desktop\mining\market_basket_project\main.py
    # So the dataset is ../Groceries_dataset.csv relative to this file
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Groceries_dataset.csv")
    
    try:
        df = load_data(dataset_path)
        print(f"Data loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_path}")
        return

    # 2. EDA
    print("Generating EDA...")
    eda_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "top_products_output.png")
    plot_top_products(df, n=15, save_path=eda_output_path)

    # 3. Preprocessing (Transaction Formation)
    print("Preprocessing transactions...")
    transactions = get_transactions(df)
    print(f"Total transactions: {len(transactions)}")
    
    # 4. Encoding
    print("Encoding transactions...")
    encoded_df = encode_transactions(transactions)
    print(f"Encoded DataFrame shape: {encoded_df.shape}")

    # 5. Modeling (Frequent Itemsets)
    min_support = 0.001 # Can be adjusted
    print(f"Finding frequent itemsets (Min Support: {min_support})...")
    frequent_itemsets = get_frequent_itemsets(encoded_df, algorithm='fpgrowth', min_support=min_support)
    print(f"Found {len(frequent_itemsets)} frequent itemsets.")

    if len(frequent_itemsets) == 0:
        print("No frequent itemsets found. Try lowering min_support.")
        return

    # 6. Association Rules
    print("Generating association rules...")
    rules = get_association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    print(f"Found {len(rules)} rules.")
    
    if len(rules) > 0:
        # Sort by lift
        rules = rules.sort_values(by="lift", ascending=False)
        output_rules_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "market_basket_results.csv")
        rules.to_csv(output_rules_path, index=False)
        print(f"Rules saved to {output_rules_path}")
        
        print("\nTop 5 Rules by Lift:")
        print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
    else:
        print("No association rules found.")

if __name__ == "__main__":
    main()
