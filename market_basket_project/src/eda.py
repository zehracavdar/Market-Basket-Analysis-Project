import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_top_products(df, n=10, save_path=None):
    """
    Plots the top n most frequent products.
    df: The original raw DataFrame (before transaction grouping) or we can simply count from the itemDescription column.
    """
    # Simply counting frequency of items in the raw dataframe
    item_counts = df['itemDescription'].value_counts().head(n)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=item_counts.values, y=item_counts.index, palette='viridis')
    plt.title(f'Top {n} Best-Selling Products')
    plt.xlabel('Frequency')
    plt.ylabel('Product')
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Top products plot saved to {save_path}")
    else:
        plt.show()

def get_item_frequency(df):
    """
    Returns the value counts of items.
    """
    return df['itemDescription'].value_counts()
