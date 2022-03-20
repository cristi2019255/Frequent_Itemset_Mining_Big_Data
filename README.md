# Frequent_Itemset_Mining_Big_Data
Frequent item set mining in context of big data <br/>

This is an analysis of how data can be sampled for frequent item set mining in order to get reasonable results in context of big data. <br/>
This analysis use the Apriori algorithm on the complete data set as a baseline for comparisons. <br/>

As sampling techniques two approaches are considered, the aproaches were presented in papers by: <br/>
1) Hannu Toivonen, Sampling Large Databases for Association <br/>
2) Riondato and Upfal (uses the PAC learning theory) (2014) <br/>

In our analysis we are interested in sample size we need to consider given each approach and the number of false negatives results we get (i.e. the itemsets that are frequent on the complete data set but are not frequent on the sample we consider). <br/>
Since the sampling is non-deterministic we run each approach 25 times and average the results in order to make reasonable conclusions. <br/>


The data sets we use are transactional data sets. <br/>
Links to the data used: <br/>
1.1) https://www.kaggle.com/code/roshansharma/market-basket-analysis/data (Market_Basket_Optimisation.csv) <br/>
1.2) https://www.kaggle.com/datasets/akalyasubramanian/dataset-for-apriori-algorithm-frequent-itemsets (Market_Basket_Optimisation.csv) (Transactions: 7500) <br/>
1)   https://www.kaggle.com/datasets/irfanasrullah/groceries (groceries-groceries.csv)    (Transactions: 9935) <br/>
2)   https://www.kaggle.com/code/oliviapointon/market-basket-analysis-apriori-algorithm/data (ItemList.csv) (Transactions: 23195) <br/>