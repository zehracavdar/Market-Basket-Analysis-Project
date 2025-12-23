from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

def get_frequent_itemsets(df, algorithm='fpgrowth', min_support=0.01):
    """
    Finds frequent itemsets using Apriori or FP-Growth.
    """
    if algorithm == 'apriori':
        frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    elif algorithm == 'fpgrowth':
        frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    else:
        raise ValueError("Algorithm must be 'apriori' or 'fpgrowth'")
    
    return frequent_itemsets

def get_association_rules(frequent_itemsets, metric="lift", min_threshold=1.0):
    """
    Generates association rules from frequent itemsets.
    """
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    return rules
