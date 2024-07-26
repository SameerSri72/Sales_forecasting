# What is this Project?
This is an elaborate project build for educational purpose which encompasses concepts from Machine learning, time series forecasting and hypothesis testing.<br>
In this project, forecasting of grocery sales of Corporación Favorita store is done.

# Dataset used:
The dataset which is publicly available on kaggle is used ([link](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data)).<br>
* file train.csv contains training data, which includes the target unit_sales by date, store_nbr, and item_nbr and a unique id to label rows.<br>
* stores.csv contains Store metadata, including city, state, type, and cluster. cluster is a grouping of similar stores.<br>
* transactions.csv contains the count of sales transactions for each date, store_nbr combination. Only included for the training data timeframe.<br>
* oil.csv contains Daily oil price. Includes values during both the train and test data timeframe.<br>
* holiday_events.csv contains the information of holidays in Ecuador.<br>
# What is being done?
This repository contains two jupyter notebooks namely sales_forecast.ipynb and sales_prophet.ipynb. In sales_forecast.ipynb exploratory data analysis, forecasting with the help of machine learning models such as polynomial regression, random forest and comparison between models is done. in the same notebook SARIMA time series model for forecasting is also build. In sales_prophet.ipynb as the name suggests facebook prophet is being for sales forecasting and holidays information is also incorporated to improve the results.  
# Important explanations and findings:
First We will start from sales_forecast.ipynb
### Data preprocessing and cleaning
This Data does not require much cleaning and preprocessing except oil price data, while checking for missing values, It is found that oil price dataset has 43 missing values. in order to judge the method to handle null values oil price variation is visualized and it is pretty evident that recent previous non null value of oil price is appropriate to fill in place of null values, this strategy is called backfill strategy.
![download](https://github.com/user-attachments/assets/6f0b9bc2-90b6-474e-825d-0a0975c94254)
### Exploratory data analysis
