from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

def read_data(file_name):
    """ Reading the data from file as a list of transactions (a transaction is a list of strings)

    Args:
        file_name (string): the file from which the transactions dataset is read

    Returns:
        list of lists: the list of transactions
    """
    transactions = []
    if file_name == './ItemList.csv':
        with open(file_name, 'r') as file:
            lines = file.readlines()
            for line in lines[1:]:                
                transactions.append(line.strip().split(',')[2:])
    else:
        data = pd.read_csv(file_name, delimiter=',', header = None)
        for i in range(len(data)): 
            transaction = []
            for j in range(len(data.values[i])):
                if not (str(data.values[i,j]) == 'nan' or str(data.values[i,j]).isdigit()):                
                    transaction.append(str(data.values[i,j]))
            transactions.append(transaction)            
    print('Data was read from file...')             
    return transactions           
    
def encode_transactions(transactions):   
    """ One hot encoding (0,1 as False,True) of the dataset of transactions

    Args:
        transactions (list of lists): the list of all the transactions

    Returns:
        (DataFrame, int, int): the dataframe with one hot encoding, the number of transactions in the dataset, the number of items in the dataset 
    """
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)    
    print('Encoding transactions done...')    
    data_size = len(df)
    print(f'Dataset size: {data_size}')
    total_nr_of_items = len(te.columns_)
    print(f'Total nr of items: {total_nr_of_items}')    
    return df, data_size, total_nr_of_items
