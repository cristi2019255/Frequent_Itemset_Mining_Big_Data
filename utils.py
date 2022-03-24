import numpy as np
from mlxtend.frequent_patterns import apriori

def compute_d_bound(df):
    """
    Computing the d-bound: the largest integer d such as the data set contains at least d different transactions of length at least d
    Args:
        df (transactions dataFrame)

    Returns:
        int: the d-bound
        
    P.S. Can be optimized in future
    """
    length_dict = {}
    # removing duplicates (in order to remain only with unique transactions) and geting the length of all transactions 
    df = df.drop_duplicates()
    df = df.sum(axis=1) # sum over rows      
    for l in df:       
       if l in length_dict.keys():
           length_dict[l] += 1                          
       else:
           length_dict[l] = 1
       
       for i in range(1,l):
            if i in length_dict.keys():
                length_dict[i] += 1                          
            else:
                length_dict[i] = 1
    keys = list(length_dict.keys())[:]
    for key in keys:
        if length_dict[key] < key:
            length_dict.pop(key)                 
    return max(length_dict.keys())


def remove_infrequent(df, sample_df, frequent_itemsets, true_support):
    """ 
    Removes the sets that are frequent on a sample but are not frequent on the complete dataset
    
    Args:
        df (dataframe): complete transactions dataframe
        sample_df (dataframe): sample dataframe
        frequent_itemsets (list of sets): the list of frequent itemsets on a sample 
        true_support (float) in (0,1]: the true support on the full data set D

    Returns:
        list of sets: the list of true frequent itemsets over the complete dataset
    """
    
    sample_supports = np.zeros(len(frequent_itemsets))
    dataset_supports = np.zeros(len(frequent_itemsets))
    data_size = len(df)    
    sample_size = len(sample_df)
    for i in range(len(frequent_itemsets)):       
        sums = df[list(frequent_itemsets[i])].sum(axis=1)         
        supp = sums[sums == len(frequent_itemsets[i])].count()                           
        dataset_supports[i] = supp / data_size 
                  
        sums = sample_df[list(frequent_itemsets[i])].sum(axis=1)         
        supp = sums[sums == len(frequent_itemsets[i])].count()    
        sample_supports[i] = supp / sample_size 
       
    true_frequent = []    
    for i in range(len(frequent_itemsets)):
        if dataset_supports[i] >= true_support:
            true_frequent.append(frequent_itemsets[i])            
            
    return true_frequent, dataset_supports, sample_supports
    
def apriori_df(data,support, show = False):
    """ Apriori algorithm for frequent item set mining

    Args:
        data (data frame): the data set or sample from which frequent item sets are mined
        support (float) in (0,1]: the support
        show (bool): flag that if set to True shows the frequent itemsets

    Returns:
        (list of sets, int): the list of frequent itemsets & the amount
    """
    frequent_itemsets = apriori(data, min_support=support, use_colnames=True)
    nr_frequent_itemsets = len(frequent_itemsets)
    
    if show:
        print('Frequent item sets found: ')
        print(frequent_itemsets)                
        print(f'Nr of frequent item sets: {nr_frequent_itemsets}')
    
    return frequent_itemsets, nr_frequent_itemsets