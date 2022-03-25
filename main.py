from experiments import RU_experiment, toivonen_experiment
from preprocesing import encode_transactions, read_data
from utils import apriori_df, compute_d_bound
from datetime import datetime

SUPPORT = 0.05
EPSILON = 0.05
DELTA = 0.0001
MIU = 0.01

#DATA_FILE = './DataSets/all_frequent.csv'
#DATA_FILE = './DataSets/Market_Basket_Optimisation.csv'
#DATA_FILE = './DataSets/groceries - groceries.csv'
#DATA_FILE = './DataSets/ItemList.csv'
DATA_FILE = './DataSets/T10I4D100K.txt'
#DATA_FILE = './DataSets/accidents.txt'

def main():  
    # Reading the data from file
    transactions = read_data(DATA_FILE)
    # Encoding the data (one-hot encoding) (boolean values)
    transactions_df, dataset_size ,total_nr_of_items = encode_transactions(transactions)        
        
    # getting the frequent itemsets on full data set    
    print(f'True support: {SUPPORT}')
    # computing d-bound
    d_bound = compute_d_bound(transactions_df)    
    print(f'd-bound: {d_bound}')
    
    
    start = datetime.now()
    true_frequent_itemsets, nr_true_frequent_itemsets = apriori_df(transactions_df, SUPPORT)
    run_time = datetime.now() - start
    print(f'Nr of true frequent itemsets: {nr_true_frequent_itemsets}')    
    print(f'Run time: {run_time}')        
    
    
    start = datetime.now()
    toivonen_experiment(
                        transactions_df=transactions_df,
                        dataset_size=dataset_size,
                        total_nr_of_items=total_nr_of_items,
                        nr_true_frequent_itemsets=nr_true_frequent_itemsets,
                        true_support=SUPPORT,
                        true_frequent_itemsets= true_frequent_itemsets,
                        epsilon= EPSILON,
                        delta = DELTA,
                        miu = MIU
                        )    
    run_time = (datetime.now() - start) / 10
    print(f'Run time: {run_time}')
    
    
    start = datetime.now()    
    RU_experiment(
                  transactions_df=transactions_df,
                  dataset_size=dataset_size,
                  nr_true_frequent_itemsets=nr_true_frequent_itemsets, 
                  true_support=SUPPORT,
                  d_bound = d_bound,
                  epsilon= EPSILON,
                  delta = DELTA
                  )    
    run_time = (datetime.now() - start) / 10
    print(f'Run time: {run_time}')
    
if __name__ == '__main__':
    main()
    