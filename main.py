from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import pandas as pd
import numpy as np
from math import log, sqrt, ceil

SUPPORT = 0.03
EPSILON = 0.1
DELTA = 0.1
MIU = 0.001
#DATA_FILE = './Market_Basket_Optimisation.csv'
DATA_FILE = './groceries - groceries.csv'

def main():
    """
    Data link: https://www.kaggle.com/code/roshansharma/market-basket-analysis/data (Market_Basket_Optimisation.csv)
               https://www.kaggle.com/datasets/akalyasubramanian/dataset-for-apriori-algorithm-frequent-itemsets (Market_Basket_Optimisation.csv)
               https://www.kaggle.com/datasets/irfanasrullah/groceries (groceries-groceries.csv)
    """
    Data = pd.read_csv(DATA_FILE, delimiter=',', header = None)
        
    # Intializing the list
    transactions = []
    # populating a list of transactions
    for i in range(len(Data)): 
        transaction = []
        for j in range(len(Data.values[i])):
            if not (str(Data.values[i,j]) == 'nan' or Data.values[i,j].isdigit()):                
                transaction.append(str(Data.values[i,j]))
        transactions.append(transaction)
            
    
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    support = SUPPORT
    data_size = len(df)
    print(f'Dataset size: {data_size}')
    
    # getting the frequent itemsets on all the data set    
    frqt_itemsets = apriori(df, min_support=support, use_colnames=True)
    print(frqt_itemsets)            
    nr_frqt_itemsets = len(frqt_itemsets)
    print(f'Nr of frequent item sets: {nr_frqt_itemsets}')
    
    epsilon = EPSILON
    delta = DELTA
    
    total_nr_of_items = len(te.columns_)
    print(f'Total nr of items: {total_nr_of_items}')
    miu = MIU
    
    # getting the frequent itemsets on Toivonen sample    
    sample_toivonen_size = min(data_size,ceil(1 / (epsilon ** 2) * (total_nr_of_items + log(2/delta))))
    min_support = support - sqrt((1/sample_toivonen_size) * log(1/miu)) / data_size    
    
    print(f'Sample Toivonen size: {sample_toivonen_size}')
    print(f'Min support: {min_support}')
    
    sample_toivonen = df.sample(frac=(sample_toivonen_size/data_size), replace=False)
    frqt_itemsets = apriori(sample_toivonen, min_support=min_support, use_colnames=True)
    print(frqt_itemsets)    
    frq_itemsets = remove_unfrequent(transactions, list(frqt_itemsets['itemsets']), support)
    #print(frq_itemsets)
    print(len(frq_itemsets))    
    fn = nr_frqt_itemsets - len(frq_itemsets)
    print(f'False negatives: {fn}')
    
    
    # RU PAC learning
    # computing d-bound
    d = get_d_bound(transactions)
    c = 0.5         
    sample_RU_size = min(data_size, ceil((4*c/(epsilon**2)) * (d + log(1/delta))))
    min_support = support - (epsilon/2) / data_size
    
    
    print(f'd-bound: {d}')
    print(f'Sample RU(PAC) size: {sample_RU_size}')
    print(f'Min support: {min_support}')
    
    sample = df.sample(frac=(sample_RU_size/data_size), replace=False)
    frqt_itemsets = apriori(sample, min_support=min_support, use_colnames=True)
    print(frqt_itemsets)
    frq_itemsets = remove_unfrequent(transactions, list(frqt_itemsets['itemsets']), support)
    #print(frq_itemsets)
    print(len(frq_itemsets))
    fn = nr_frqt_itemsets - len(frq_itemsets)
    print(f'False negatives: {fn}')
    
        
    
def get_d_bound(D):
    ## Computing the d-bound     
    L = {}
    for transaction in D:
       l = len(transaction) 
       if l in L.keys():
           L[l] += 1                          
       else:
           L[l] = 1
       
       for i in range(1,l):
        if i in L.keys():
            L[i] += 1                          
        else:
            L[i] = 1
    keys = list(L.keys())[:]
    for key in keys:
        if L[key] < key:
            L.pop(key)                 
    return max(L.keys())

def remove_unfrequent(D, frqt_itemsets, true_support):
    supports = np.zeros(len(frqt_itemsets))
    data_size = len(D)
    for transaction in D:
        for i in range(len(frqt_itemsets)):
            if frqt_itemsets[i].issubset(transaction):
                supports[i] += 1 / data_size           
    frq = []
    for i in range(len(frqt_itemsets)):
        if supports[i] >= true_support:
            frq.append(frqt_itemsets[i])        
    
    return frq
    
if __name__ == '__main__':
    main()
    