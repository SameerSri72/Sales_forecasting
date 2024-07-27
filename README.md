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
## Data preprocessing and cleaning
This Data does not require much cleaning and preprocessing except oil price data, while checking for missing values, It is found that oil price dataset has 43 missing values. in order to judge the method to handle null values oil price variation is visualized and it is pretty evident that recent previous non null value of oil price is appropriate to fill in place of null values, this strategy is called backfill strategy.
![download](https://github.com/user-attachments/assets/6f0b9bc2-90b6-474e-825d-0a0975c94254)
## Exploratory data analysis
* Total sales per family is shown below, It is clearly visible that GroceryI, Produce and beverages are most sold families.
![download](https://github.com/user-attachments/assets/ff0a41a5-7e4f-4d4e-a59b-6c3f1d7beba1)

* If we want to know the about the cities where sales are high.
![download](https://github.com/user-attachments/assets/0b2e0ba9-3899-4fce-a925-25a9250f11c7)
* Average sales in States can be seen below
![download](https://github.com/user-attachments/assets/1df681a7-d5ce-4bfc-a7c1-1895a9cf6f69)
* Correlation between sales, number of transactions and oil prices are given below, decent positive correlation exist betweeen no. of transactions and sales. There is insignificant correlation between oil prices and sales.
![download](https://github.com/user-attachments/assets/1da914cd-be23-44ff-a403-4c6c034a811d)
## Hypothesis Testing:
### Null Hypothesis: The promotional activities do not have a significant impact on store sales for Corporation Favorita.<br>**Alternate Hypo (H1)**: The promotional activities have a significant impact on store sales for Corporation Favorita.<br>
Dataset contains a column named *onpromotion* it takes value 0 when promotional activities are not done for the respective family and 1 otherwise. Two sample t test is used to compare the sales in promotion and non promotion, for p value less than 0.05, Null hypothesis is rejected. Result is Null hypothesis got rejected because of very less p value, hence promotional activities have significant impact on store sales.
![download](https://github.com/user-attachments/assets/25e156b1-dae1-4508-a018-c6739d8b4fb7)

**Judging the impact of earthquake occured on 16th april 2016 on store sales**<br>
### Null Hypo: Earthquake does not have a significant impact on sales.<br>**Alternate hypo**: Earthquake have a significant impact on sales.<br>
As distribution of sales is nowhere close to normal so we can not use t test to compare sales before and after earthquake happened hence we use Mann-Whitney U test which is a non parametric equvivalent of t test. the result is Null hypothesis getting rejected hence Earthquake had a significant impact on sales.<br>
![download](https://github.com/user-attachments/assets/20a77715-bb1c-4ed8-9471-bfb221e5b3aa)
