from experiments import RU_experiment, toivonen_experiment
from preprocesing import encode_transactions, read_data
from utils import apriori_df

SUPPORT = 0.02
#DATA_FILE = './DataSets/Market_Basket_Optimisation.csv'
#DATA_FILE = './DataSets/groceries - groceries.csv'
DATA_FILE = './DataSets/ItemList.csv'


def main():  
    # Reading the data from file
    transactions = read_data(DATA_FILE)
    # Encoding the data (one-hot encoding) (boolean values)
    transactions_df, dataset_size ,total_nr_of_items = encode_transactions(transactions)        
        
    # getting the frequent itemsets on full data set    
    print(f'True support: {SUPPORT}')
    true_frequent_itemsets, nr_true_frequent_itemsets = apriori_df(transactions_df, SUPPORT)
    print(f'Nr of true frequent itemsets: {nr_true_frequent_itemsets}')
        
    toivonen_experiment(transactions=transactions,
                        transactions_df=transactions_df,
                        dataset_size=dataset_size,
                        total_nr_of_items=total_nr_of_items,
                        nr_true_frequent_itemsets=nr_true_frequent_itemsets,
                        true_support=SUPPORT,
                        true_frequent_itemsets= true_frequent_itemsets)    
    
    RU_experiment(transactions=transactions,
                  transactions_df=transactions_df,
                  dataset_size=dataset_size,
                  nr_true_frequent_itemsets=nr_true_frequent_itemsets, 
                  true_support=SUPPORT,
                  true_frequent_itemsets= true_frequent_itemsets)    
        
if __name__ == '__main__':
    main()
    