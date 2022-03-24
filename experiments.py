import numpy as np
from math import log, sqrt, ceil
from utils import apriori_df, compute_d_bound, remove_infrequent


def toivonen_experiment(transactions, transactions_df, dataset_size, total_nr_of_items, nr_true_frequent_itemsets,  true_support, true_frequent_itemsets, epsilon, delta, miu):
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
    support = true_support - sqrt((1/(2 *sample_toivonen_size)) * log(1/miu)) 
    
    print(f'Sample Toivonen size: {sample_toivonen_size}')
    print(f'Support on sample: {support}')
    
    
    false_positives = []
    false_negatives = []
    
    for _ in range(10):
        sample_toivonen = transactions_df.sample(frac=(sample_toivonen_size/dataset_size), replace=False)
    
        toivonen_frequent_itemsets, nr_toivonen_frequent_itemsets = apriori_df(sample_toivonen, support)
    
        frq_itemsets, sample_supports = remove_infrequent(transactions_df, list(toivonen_frequent_itemsets['itemsets']), true_support)
        
        dataset_supports = list(true_frequent_itemsets[true_frequent_itemsets['itemsets'].isin(frq_itemsets)]['support'])        
        try: 
            check_guarantees_Toivonen(dataset_supports, sample_supports, sample_threshold= support, nr_of_items=total_nr_of_items, epsilon=epsilon, delta=delta, miu=miu)        
        except Exception as e:
            print(e)
        
        false_negatives.append(nr_true_frequent_itemsets - len(frq_itemsets))
        false_positives.append(nr_toivonen_frequent_itemsets - len(frq_itemsets))
        
    print(f'Nr of Toivonen frequent itemsets: {nr_toivonen_frequent_itemsets}')    
    print(f'False negatives: {false_negatives}')
    print(f'False positives: {false_positives}')
    fn_avg, fn_std = np.mean(false_negatives), np.std(false_negatives)
    fp_avg, fp_std = np.mean(false_positives), np.std(false_positives)
    print(f'False negatives mean (standard dev): {fn_avg} ({fn_std})')
    print(f'False positives mean (standard dev): {fp_avg} ({fp_std})')
    
def RU_experiment(transactions, transactions_df, dataset_size, nr_true_frequent_itemsets,true_support, true_frequent_itemsets, epsilon, delta):
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
    support = true_support - (epsilon/2)
    
    
    print(f'd-bound: {d}')
    print(f'Sample RU(PAC) size: {sample_RU_size}')
    print(f'Support on sample: {support}')
        
    false_positives = []
    false_negatives = []
    
    for _ in range(10):    
        sample_RU = transactions_df.sample(frac=(sample_RU_size/dataset_size), replace=False)    
        RU_frequent_itemsets, nr_RU_frequent_itemsets = apriori_df(sample_RU, support)            
        frq_itemsets, sample_supports = remove_infrequent(transactions_df, list(RU_frequent_itemsets['itemsets']), true_support)
                
        dataset_supports = list(true_frequent_itemsets[true_frequent_itemsets['itemsets'].isin(frq_itemsets)]['support'])        
        try: 
            check_guarantees_RU(dataset_supports, sample_supports, epsilon=epsilon, delta=delta)        
        except Exception as e:
            print(e)        
        
        false_negatives.append(nr_true_frequent_itemsets - len(frq_itemsets))
        false_positives.append(nr_RU_frequent_itemsets - len(frq_itemsets))
    
    print(f'Nr of RU frequent itemsets: {nr_RU_frequent_itemsets}')
    
    print(f'False negatives: {false_negatives}')
    print(f'False positives: {false_positives}')
    fn_avg, fn_std = np.mean(false_negatives), np.std(false_negatives)
    fp_avg, fp_std = np.mean(false_positives), np.std(false_positives)
    print(f'False negatives mean (standard dev): {fn_avg} ({fn_std})')
    print(f'False positives mean (standard dev): {fp_avg} ({fp_std})')    
    
    
def check_guarantees_RU(dataset_supports, sample_supports, epsilon, delta):
    assert(len(dataset_supports) == len(sample_supports))
    nr_of_errors = 0
    for i in range(len(dataset_supports)):
        if abs(dataset_supports[i] - sample_supports[i]) > epsilon / 2:
            nr_of_errors += 1
    
    if (nr_of_errors/len(dataset_supports) >= delta):
        raise Exception('Something is wrong!!!')     

def check_guarantees_Toivonen(dataset_supports, sample_supports, sample_threshold, nr_of_items, epsilon, delta, miu ):
    assert(len(dataset_supports) == len(sample_supports))
    nr_of_errors = 0
    nr_of_errors_on_sample = 0
    for i in range(len(dataset_supports)):
        if abs(dataset_supports[i] - sample_supports[i]) > epsilon:
            nr_of_errors += 1
        if sample_supports[i] >= sample_threshold:
            nr_of_errors_on_sample += 1
    
    if (nr_of_errors/len(dataset_supports) >= delta):
        raise Exception('Something is wrong!!!')     
    if (nr_of_errors_on_sample/ (2** nr_of_items) >= miu):
        raise Exception('Something is wrong!!!')     
   