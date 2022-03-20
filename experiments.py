import numpy as np
from math import log, sqrt, ceil
from utils import apriori_df, compute_d_bound, remove_infrequent


EPSILON = 0.1
DELTA = 0.01
MIU = 0.001

def toivonen_experiment(transactions, transactions_df, dataset_size, total_nr_of_items, nr_true_frequent_itemsets,  true_support, true_frequent_itemsets, epsilon = EPSILON, delta = DELTA, miu = MIU):
    """
    Getting the frequent itemsets on a sample with size calculated from Toivonen approach
    Because the sampling is non-deterministic take the average results over 25 runs

    Args:
        transactions (lists of transactions)
        transactions_df (dataFrame)
        dataset_size (int) 
        total_nr_of_items (int)
        nr_true_frequent_itemsets (int)
        epsilon (float, optional):  Defaults to EPSILON.
        delta (float, optional):  Defaults to DELTA.
        miu (float, optional): Defaults to MIU.
        true_support (float, optional): Defaults to SUPPORT.
    """
    
    sample_toivonen_size = min(dataset_size, ceil(1 / (epsilon ** 2) * (total_nr_of_items + log(2/delta))))
    support = true_support - sqrt((1/sample_toivonen_size) * log(1/miu)) / dataset_size    
    
    print(f'Sample Toivonen size: {sample_toivonen_size}')
    print(f'Support on sample: {support}')
    
    
    false_positives = []
    false_negatives = []
    
    for _ in range(25):
        sample_toivonen = transactions_df.sample(frac=(sample_toivonen_size/dataset_size), replace=False)
    
        toivonen_frequent_itemsets, nr_toivonen_frequent_itemsets = apriori_df(sample_toivonen, support)
    
        frq_itemsets, sample_supports = remove_infrequent(transactions, list(toivonen_frequent_itemsets['itemsets']), true_support)
        
        dataset_supports = list(true_frequent_itemsets[true_frequent_itemsets['itemsets'].isin(frq_itemsets)]['support'])        
        try: 
            check_guarantees(dataset_supports, sample_supports)        
        except Exception as e:
            print(e)
        
        false_negatives.append(nr_true_frequent_itemsets - len(frq_itemsets))
        false_positives.append(nr_toivonen_frequent_itemsets - len(frq_itemsets))
        
    #print(f'False negatives: {false_negatives}')
    #print(f'False positives: {false_positives}')
    fn_avg, fn_std = np.mean(false_negatives), np.std(false_negatives)
    fp_avg, fp_std = np.mean(false_positives), np.std(false_positives)
    print(f'False negatives mean (standard dev): {fn_avg} ({fn_std})')
    print(f'False positives mean (standard dev): {fp_avg} ({fp_std})')
    
def RU_experiment(transactions, transactions_df, dataset_size, nr_true_frequent_itemsets,true_support, true_frequent_itemsets, epsilon = EPSILON, delta = DELTA):
    """
    Getting the frequent itemsets on a sample with size calculated from Riondato and Upfal approach
    Because the sampling is non-deterministic take the average results over 25 runs

    Args:    
        transactions (lists of transactions)
        transactions_df (dataFrame)
        dataset_size (int)         
        nr_true_frequent_itemsets (int)
        true_support (float)
        epsilon (float, optional):  Defaults to EPSILON.
        delta (float, optional):  Defaults to DELTA.        
    """
    
    # computing d-bound
    d = compute_d_bound(transactions_df)
    
    c = 0.5         
    sample_RU_size = min(dataset_size, ceil((4*c/(epsilon**2)) * (d + log(1/delta))))
    support = true_support - (epsilon/2) / dataset_size
    
    
    print(f'd-bound: {d}')
    print(f'Sample RU(PAC) size: {sample_RU_size}')
    print(f'Support on sample: {support}')
    
    
    false_positives = []
    false_negatives = []
    
    for _ in range(25):    
        sample_RU = transactions_df.sample(frac=(sample_RU_size/dataset_size), replace=False)    
        RU_frequent_itemsets, nr_RU_frequent_itemsets = apriori_df(sample_RU, support)            
        frq_itemsets, sample_supports = remove_infrequent(transactions, list(RU_frequent_itemsets['itemsets']), true_support)
                
        dataset_supports = list(true_frequent_itemsets[true_frequent_itemsets['itemsets'].isin(frq_itemsets)]['support'])        
        try: 
            check_guarantees(dataset_supports, sample_supports)        
        except Exception as e:
            print(e)        
        
        false_negatives.append(nr_true_frequent_itemsets - len(frq_itemsets))
        false_positives.append(nr_RU_frequent_itemsets - len(frq_itemsets))
    
    
    #print(f'False negatives: {false_negatives}')
    #print(f'False positives: {false_positives}')
    fn_avg, fn_std = np.mean(false_negatives), np.std(false_negatives)
    fp_avg, fp_std = np.mean(false_positives), np.std(false_positives)
    print(f'False negatives mean (standard dev): {fn_avg} ({fn_std})')
    print(f'False positives mean (standard dev): {fp_avg} ({fp_std})')
    
    
def check_guarantees(dataset_supports, sample_supports, epsilon = EPSILON, delta = DELTA):
    assert(len(dataset_supports) == len(sample_supports))
    nr_of_errors = 0
    for i in range(len(dataset_supports)):
        if abs(dataset_supports[i] - sample_supports[i]) >= epsilon:
            nr_of_errors += 1
    
    if (nr_of_errors/len(dataset_supports) >= delta):
        raise Exception('Something is wrong!!!')        