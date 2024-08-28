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
![download](https://github.com/SameerSri72/Sales_forecasting/blob/main/images/sale1.png)

* If we want to know the about the cities where sales are high.
![download](https://github.com/SameerSri72/Sales_forecasting/blob/main/images/avgsalcity.png)
* Average sales in States can be seen below
![download](https://github.com/SameerSri72/Sales_forecasting/blob/main/images/avgsalestate.png)
* Correlation between sales, number of transactions and oil prices are given below, decent positive correlation exist betweeen no. of transactions and sales. There is insignificant correlation between oil prices and sales.
![download](https://github.com/SameerSri72/Sales_forecasting/blob/main/images/correl.png)
## Hypothesis Testing:
### Null Hypothesis: The promotional activities do not have a significant impact on store sales for Corporation Favorita.<br>**Alternate Hypo (H1)**: The promotional activities have a significant impact on store sales for Corporation Favorita.<br>
Dataset contains a column named *onpromotion* it takes value 0 when promotional activities are not done for the respective family and 1 otherwise. Two sample t test is used to compare the sales in promotion and non promotion, for p value less than 0.05, Null hypothesis is rejected. Result is Null hypothesis got rejected because of very less p value, hence promotional activities have significant impact on store sales.
![download](https://github.com/SameerSri72/Sales_forecasting/blob/main/images/hypo1.png)

**Judging the impact of earthquake occured on 16th april 2016 on store sales**<br>
### Null Hypo: Earthquake does not have a significant impact on sales.<br>**Alternate hypo**: Earthquake have a significant impact on sales.<br>
As distribution of sales is nowhere close to normal so we can not use t test to compare sales before and after earthquake happened hence we use Mann-Whitney U test which is a non parametric equvivalent of t test. the result is Null hypothesis getting rejected hence Earthquake had a significant impact on sales.<br>
![download](https://github.com/SameerSri72/Sales_forecasting/blob/main/images/hypo2.png)

**Monthly, weakly and yearly sales are represented below**<br>
![download](https://github.com/SameerSri72/Sales_forecasting/blob/main/images/month_sale_trend.png)<br>
![download](https://github.com/SameerSri72/Sales_forecasting/blob/main/images/avgsalebyweek.png)<br>
![download](https://github.com/SameerSri72/Sales_forecasting/blob/main/images/yearlsalestrend.png)<br>
## Forecasting through Machine Learning models:
### Feature Engineering:
To make the dataset suitable for machine learning models following steps are taken (details can be seen in jupyetr notebook):
* Unnecessary Columns like date,id,  locale, locale_name, description, store_type, transferred, state from merged dataset.
* 7 categories are defined such as food_families, home_families, clothing_families etc to categorize the families present in data.
* sales, no. of transactions and oil price have numerical values so standard scaling is done to nullify the different units.
* Categorical Variables are one hot encoded.
## Models:
### Polynomial regression
With degree equals to 1, only intercept term is added to data so it is a linear regression to be precise. MSE turned out to be 0.70
### Random forest regression
Parammeter n_estimator is taken as 100 so random forest will use 100 decision trees, results for this are RMSLE = 0.22,	RMSE = 0.71,	MSE = 0.51,	MAE=0.24.
### Histogram based gradient boost
Results are Model	RMSLE = 0.22,	RMSE = 0.62,	MSE = 0.39,	MAE = 0.26
## Time series forecasting models:
Now We will build Sarima model for forecasting, Data need to be transformed in the following way for this model to apply:<br>
At first total sales for each family is found and total sales is grouped on family. we take top 10 family where sales are high for the prediction. these families are:
1. GROCERY I        
2. BEVERAGES        
3. PRODUCE          
4. CLEANING         
5. DAIRY            
6. BREAD/BAKERY     
7. POULTRY          
8. MEATS            
9. PERSONAL CARE    
10. DELI<br>
to predict for each family, we need to build seperate models for each family, here we only build for Grocery I and for the other categories same approach can be taken.
AFter filtering out the daily data for sales of Grocery I, it is plotted to observe for trend, seasonality etc.
![download](https://github.com/SameerSri72/Sales_forecasting/blob/main/images/grocery_saletrend.png) <br>
As it is visible from the plot that at the starting of each year sale is very less and between the year 2016 and 2017 there is a sudden peak in the sale as well, we have the information regarding occurence of earth on 16 april 2016, so this sudden increment in sale is an effect of this as being tested in hypothesis testing section. slight upward trend is there but it is not very significant.
### Testing stationarity of data using Augmented dickey fuller (ADF) test:
Stationarity means that the statistical properties of the series like mean, variance, and autocorrelation are constant over time. A stationary time series is crucial for many time series forecasting models as they assume the underlying data is stationary.<br>
**Null Hypothesis (H0):** The time series has a unit root, meaning it is non-stationary.<br>
**Alternate Hypothesis (H1):** The time series does not have a unit root, meaning it is stationary.<br>
If the p-value is less than a chosen significance level (commonly 0.05), the null hypothesis is rejected, indicating the series is stationary.<br>
After performing the test the value of test statistic is found out to be -3.45 and p value is 0.009. Hence Null Hypothesis is rejected which leads to conclusion that data is stationary.
### ACF and PACF plots:
ACF and PACF plot for the grocery sales is given below:<br>
![download](https://github.com/SameerSri72/Sales_forecasting/blob/main/images/acf1.png)<br>
![download](https://github.com/SameerSri72/Sales_forecasting/blob/main/images/pacf1.png)<br>

ACF plot reveals the periodic pattern with period 7, which leads to conclusion that weekly seasonality is present in the data, which is also visible rfom PACF plot as well.<br>
### Building SARIMA model:
SARIMA model is being built by taking the following parameters:<br>
1. p_range = range(1, 9) (for the Autoregressive part)
2. d = 0 
3. q = 0 (no dependency on previous error terms so Moving Average part is 0)
4. P = 1 (current sale depends upon the previous 7th day)
5. D_range = range(1, 4) 
6. Q = 0 (Sale does not depend upon the error of previous 7th day)
7. s = 7 (period of seasonality)<br>
Based on AIC criteria the best model parameters are found out to be p=6.0, d=0.0, q=0.0, P=1.0, D=1.0, Q=0.0, s=7.0, with AIC=38882.61 <br>
After prediction the RMSLE value turn out to be 0.112, interpretation of RMSLE = 0.112 means on an average the log of the predicted values are 11.2 % off from the log of actual values.To visually see the model performance actual vs predicted value plot is being made.<br>
![download](https://github.com/SameerSri72/Sales_forecasting/blob/main/images/saractvspred.png)<br>
As we can see initially model performs satisfactorily but at the end not able to adapt as desired hence the gap between actual and predicted values.
### Using Auto ARIMA:
Using Auto ARIMA with parameters seasonal= True and m=7(weekly seasonality), Auto ARIMA gives model ARIMA(5,1,1)(2,0,2)(7) so values of p,d,q,P,D,Q and s are 5,1,1,2,0,2 and 7.<br>
The RMSLE for this model turned out to be 0.10 and to visually see the model performance here also actual vs predicted plot is made.<br>
![download](https://github.com/SameerSri72/Sales_forecasting/blob/main/images/autoarima.png)<br>
Here also model performs satisfactorily for the initial data and at the end specially around time 15-20 and 25-30, there is a huge gap between actula and predicted values.<br>

# Using Facebook Prophet for forecasting:
## Total sales across all stores familywise:
### Automotive:
![download](https://github.com/SameerSri72/Sales_forecasting/blob/main/images/automotive.png)
### Beverages:
![download](https://github.com/SameerSri72/Sales_forecasting/blob/main/images/bevg.png)
### Books:
![download](https://github.com/SameerSri72/Sales_forecasting/blob/main/images/books.png)
### Dairy:
![download](https://github.com/SameerSri72/Sales_forecasting/blob/main/images/dairy.png)
for other families plot can be seen in sale_prophet.ipynb. As it is visible that different families give different kind of plots for several plots like dairy, automotive, grocery etc sales in the starting of the year is very low, for books sales is 0 till 2016. similarly other plots can also be interpreted.<br>
for certain families data is negligible before 2015, hence data after 2015 will be used here for model building.
### Data cleaning:
Following steps are taken for the data cleaning:
1. Only those columns are kept in the dataframe where avg sales is greater than 1000.
2. Removing outliers by using Z score which means we calculate z score for each value in the column, z score indicates how many std dev away the value lies from the mean. we put the condition z>2.7 because 99.7% data lies under 3 std dev. Any value which lies outside 2.7 is considered as an outlier and removed.<br>
After cleaning data for different families is plotted again, some of the plots are shown below:<br>
### Beverages:
![download](https://github.com/SameerSri72/Sales_forecasting/blob/main/images/bevgs2.png)
### Dairy:
![download](https://github.com/SameerSri72/Sales_forecasting/blob/main/images/dairy2.png)
### Grocery II:
![download](https://github.com/SameerSri72/Sales_forecasting/blob/main/images/grocery2.png)<br>
The one advantage in fb prophet is we can utilize holiday data, we can define a window around holiday and incorporate into our model which leads to better forcasting. plots for each column is in the notebook, some references are given below: <br>
![download](https://github.com/SameerSri72/Sales_forecasting/blob/main/images/fb1.png)<br>
![download](https://github.com/SameerSri72/Sales_forecasting/blob/main/images/fb2.png)<br>
As one can see in the first plot, less sales at the starting of each year is captured by the model.(black dot represents actual data and blue lines represent prediction)




 
