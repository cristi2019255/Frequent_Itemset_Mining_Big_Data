from turtle import color
import numpy as np
from math import log, sqrt, ceil
from utils import apriori_df, compute_d_bound, remove_infrequent
import matplotlib.pyplot as plt


def toivonen_experiment(transactions_df, dataset_size, total_nr_of_items, nr_true_frequent_itemsets,  true_support, true_frequent_itemsets, epsilon, delta, miu):
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
    
        frq_itemsets, dataset_supports, sample_supports = remove_infrequent(transactions_df, sample_toivonen, list(toivonen_frequent_itemsets['itemsets']), true_support)
                
        try: 
            check_guarantees_Toivonen(dataset_supports, sample_supports, sample_threshold= support, epsilon=epsilon, delta=delta, miu=miu)        
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
    
def RU_experiment(transactions_df, dataset_size, nr_true_frequent_itemsets,true_support, d_bound, epsilon, delta):
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
    
    
    c = 0.5         
    sample_RU_size = min(dataset_size, ceil((4*c/(epsilon**2)) * (d_bound + log(1/delta))))
    support = true_support - (epsilon/2)
            
    print(f'Sample RU(PAC) size: {sample_RU_size}')
    print(f'Support on sample: {support}')
        
    false_positives = []
    false_negatives = []
    
    for _ in range(10):    
        sample_RU = transactions_df.sample(frac=(sample_RU_size/dataset_size), replace=True)    
        RU_frequent_itemsets, nr_RU_frequent_itemsets = apriori_df(sample_RU, support)            
        frq_itemsets, dataset_supports, sample_supports = remove_infrequent(transactions_df, sample_RU, list(RU_frequent_itemsets['itemsets']), true_support)                        
        try: 
            check_guarantees_RU(dataset_supports, sample_supports, epsilon=epsilon, delta=delta, true_support=true_support)        
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
    
    
def check_guarantees_RU(dataset_supports, sample_supports, epsilon, delta, true_support):
    assert(len(dataset_supports) == len(sample_supports))
    nr_of_errors_approximation = 0
    nr_of_errors_almost_frequent = 0
    for i in range(len(dataset_supports)):
        if abs(dataset_supports[i] - sample_supports[i]) > epsilon / 2:
            nr_of_errors_approximation += 1
        if (dataset_supports[i] < true_support - epsilon):
            nr_of_errors_almost_frequent += 1
            
    if (nr_of_errors_approximation/len(dataset_supports) >= delta):
        raise Exception('Something is wrong!!!')     
    if (nr_of_errors_almost_frequent/len(dataset_supports) >= delta):
        raise Exception('Something is wrong!!!')     

def check_guarantees_Toivonen(dataset_supports, sample_supports, sample_threshold, epsilon, delta, miu ):
    assert(len(dataset_supports) == len(sample_supports))
    nr_of_errors = 0
    nr_of_errors_on_sample = 0
    for i in range(len(dataset_supports)):
        if abs(dataset_supports[i] - sample_supports[i]) > epsilon:
            nr_of_errors += 1
        if sample_supports[i] < sample_threshold:
            nr_of_errors_on_sample += 1
    
    if (nr_of_errors/len(dataset_supports) >= delta):
        raise Exception('Something is wrong!!!')     
    if (nr_of_errors_on_sample/ len(dataset_supports) >= miu):
        raise Exception('Something is wrong!!!')     
    
def plot_sizes(total_nr_of_items, d_bound, dataset_size, delta1 = 10 ** (-4), delta2 = 0.01):
    epsilons = np.arange(10 ** (-4), 0.2, 10 ** (-3))    
    sample_toivonen_sizes1 = [min(dataset_size, ceil(1 / (epsilon ** 2) * (total_nr_of_items + log(2/delta1)))) for epsilon in epsilons]
    sample_RU_sizes1 = [min(dataset_size, ceil(((4*0.5)/(epsilon**2)) * (d_bound + log(1/delta1)))) for epsilon in epsilons]
    sample_toivonen_sizes2 = [min(dataset_size, ceil(1 / (epsilon ** 2) * (total_nr_of_items + log(2/delta2)))) for epsilon in epsilons]
    sample_RU_sizes2 = [min(dataset_size, ceil(((4*0.5)/(epsilon**2)) * (d_bound + log(1/delta2)))) for epsilon in epsilons]
    plt.title('Sample size for epsilon values')
    plt.plot(epsilons, sample_toivonen_sizes1, color = 'green')
    plt.plot(epsilons, sample_RU_sizes1, color = 'b')
    plt.plot(epsilons, sample_toivonen_sizes2, '--', color = 'red')
    plt.plot(epsilons, sample_RU_sizes2, '--', color = 'orange')
    
    plt.legend(['Toivonen, delta = 10^(-4)', 'Riondato & Upfal, delta = 10^(-4)', 'Toivonen, delta = 0.01', 'Riondato & Upfal, delta = 0.01'])
    plt.xlabel('epsilon')
    plt.ylabel('Sample size')    
    plt.show()