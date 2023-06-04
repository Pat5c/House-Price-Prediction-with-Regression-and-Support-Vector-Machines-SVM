# House Price Prediction with Regression and Support Vector Machines (SVM)

# Contents

1.	Abstract …………………………………………………………………………………………………………………………. 3

2.	Introduction …………………………………………………………………………………………………………………… 4

3.	Methods ………………………………………………………………………………………………………………………… 6

4.	Results ……………………………………………………………………………………………………………………………. 13

5.	Discussion & Conclusions ……………………………………………………………………………………………….. 24

6.	Bibliography ……………………………………………………………………………………………………………………. 26

7.	Appendices ……………………………………………………………………………………………………………………… 27

# 1.	Abstract 
This research aims to predict house prices for houses located in the US. There are some factors that can influence the price of a house such us the location, size or the physical conditions; predicting the price of houses can help both buyers and sellers determine the best time to purchase a house and which qualities to look at, when making the decision.
This project is finalized into finding the best model that can predict the housing price. As continuous, house prices will be predicted with various regression techniques including Linear and Polynomial regression models and a Support Vector Machines (SVM) model.
The following features: Bed, Bath, Building Area, Longitude and Latitude have been taken into consideration to make predictions. These variables have been analysed, interchanged, or omitted (according to the results) in order to find the best performing model.
As it will be shown, the observed results indicate that combining the variables by using a general linear model will not create significant results (with a R-squared for the prediction model on the full data set at low as 0.0078).
The research will also study the effect of these variables on the regression analysis and predictive model, capturing the changes in the models’ results when using the full dataset (inclusive of outliers) and a cleaned and smaller dataset (without outliers).
A detailed investigation on how the quality of the input data affects the final model will be done with the use of train and validation data, on the best performing models.
The results will show that the polynomial regression model and the SVM model, will produce slightly improved results on the dataset without outliers, creating R-Squared values for the prediction model on the validation data set of 0.1280 and 0.2500 respectively.

# 2.	Introduction 
Buying a house is easily the biggest and most important financial investment any person may do during their life time. For real estate agencies, knowing the best price at which selling houses and for buyers, knowing the best values at which acquiring houses, is of paramount importance.
Predicting the price of houses is both a challenge and a huge business, with companies such as Zoopla or Right Move offering online valuations of houses. In the US, for example, a similar company called Zillow has launched a $1 million competition called the Zillow Prize, to improve their house price evaluations, as argued in O'Farrell, (2014).
The dataset presented in this study, shows a real estate dataset from all around the United States. This cleaned Csv file comprises 9,992 records relevant to Real Estate investors, agents, and buyers. The full dataset has been collected on the 27th June 2021, and it can be downloaded for free from the Zenrows website (ZenRows, (2021)). I chose this dataset because it is clear, comprehensive and quite big. Many numerical values are present which make it suitable for model building and regression analysis.
Prior academic work with different datasets has been completed on how to predict house prices, the works of Yu and Wu, (2016); Bharathi et al, (2019); again Wu, (2017) in “Housing Price prediction Using Support Vector Regression” and finally O'Farrell, (2014) are all good examples.
In this project, after initially analysing the suitability of the dataset for linear regression, by double checking the assumptions for regression, we will analyse the dataset by removing any apparent outliers or high-leverage / high-influence data points in the dataset (we will take into consideration only house prices below the value of 500,000$).
After this removal and after normalizing the datasets for better model performance, we will then fit both datasets to linear and polynomial regression models.
We will be using the caret package in R and the “preProcess” function for carrying out the data normalization steps on both datasets (on the full and on the reduced one); the output of the normalization will show that all the numerical variables have been standardized with a mean value of zero.
In regards to the regressions models used in this project, it is known that regression analysis helps to understand how the value of the dependent variable changes when any one of the independent variables is varied, while the other independent variables are held fixed. 
Both linear and polynomial regression will be calculated in the project: linear regression models use a straight line, while the latter is a technique we can use when the relationship between a predictor variable and a response variable is nonlinear.
As it will be shown in the project, the polynomial regression model is better fitted on both datasets, but we will still build the linear regression model in order to make comparisons and analyse the suitability of the model to the full dataset and to the cleaned dataset (without outliers).
To better improve the models, backward elimination based on the summary statistics will be used to remove the predictor with the largest p-value (> 0.05): the p-Values are very important because we can consider a model to be statistically significant only when both the general and the individual predictor variables p-Values, are less that the pre-determined statistical significance level of 0.05.
Additionally, ANOVA analysis using with the anova() function (which performs an analysis of variance ANOVA, using a F-test) will be implemented to help us analyse the error of all the implemented models.
While it seems more reasonable to perform regression since house prices are continuous, we will also perform a SVM analysis: Support Vector Machine is a supervised learning method that can be used for both regression and classification problems. The “e1071” package in R provides an “svm” function to build support vector machines model for regression problems.
As it will be shown in the project, this model will provide an overall better performance.
Nevertheless, all of the best performing models will be predicted, to test and find the better fitted and performing one. Calculating the value of R-squared for the predicted models on the data set, will help us evaluate the goodness of fit of models: higher R-squared means a better result, with 1 being the best.
Lastly, dividing the dataset into training and validation datasets will help us both train the best performing models and test the accuracy of the models, until we are satisfied. 


# 3.	Methods
The data utilized for this project shows the prices and features of residential houses collected from all around the United States and obtained from Zenrows website (ZenRows, (2021)).
The raw and complete dataset consists of 47 house features and 10,000 records of house prices relevant to Real Estate investors, agents and buyers. The dataset has been collected on the 27th June 2021, and it can be downloaded for free from the website (cleaning and structure of the full raw dataset has been reported in the Appendices section of this report.
The dataset presents features in a variety of formats: it has numerical data such as prices and numbers of bedrooms, bathrooms, longitude and latitude as well as categorical features such as the address, detail of the house URL or the status of the house etc.
For the purpose of this analysis and in order to make the data usable for models, some data cleaning has been undertaken; the dataset’s features have been reduced to 11: ID, Price, Address, City, Zipcode, Number of Bedrooms/Bathrooms, Building surface area and Latitude/Longitude and appropriate type has been allocated to each column.
To not influence the models and to achieve a complete and clean dataset, ready for modelling, the number of rows have been reduced to 9,992 after removing all of the “NULL” Price rows.
Additionally, NAs have been taken care of by replacing the word “Not provided” to the column Address and “0” to the column Zipcode. Finally, for the numerical columns, which are beds, baths, Building area, Latitude and longitude, every NA cell has been replaced with the column’s respective mean.

The first step of this data experiment, is to install and load all of the required libraries, as listed in the CW2 script (is advisable to run one line at the time).
After uploading our cleaned dataset called “us-cities-real-estate-sample-zenrows.csv" and setting the correct datatypes, we are now ready to analyse the dataset to check whether all the assumptions to apply linear regression to the Dataset are met (please, when uploading the file kindly remember to replace “C:/Users/patie/OneDrive/Documents/" with your own directory).
The first assumption checked, is the one of independence of observations to make sure they aren’t too highly correlated. This is done by producing a table of Pearson’s correlation coefficients rounded to two decimal places.
The second assumption checked, is the one of normality: we are checking whether the dependent variable follows a normal distribution, with the use of an histogram.
Next, the assumption of linearity is checked by comparing ggplot graphs for better visualization: the dependent variable (Price) is compared to the independent variables, in particular: Price versus Bed is checked first, following by Price versus Baths, Price versus Building Area and finally Price versus Latitude and Longitude.
Lastly, the relationship between all of the independent variables (Beds, Baths, Building Area, Latitude and Longitude) is checked with the use of a scatterplot matrix using the “pairs()” function.

As the assumptions are not completely met with the full dataset, the exact same analysis is then made to a reduced dataset: this new dataset, called “new_df” includes only house prices below price 500,000$.
The aim of this reduction is to remove those apparent outliers or high-leverage or high-influence data points to double check how the dataset behaves and to eventually refit the regression to hopefully register an overall improvement in the model performance.
The new dataset without outliers is made up of 8,458 observations.

We now want to standardize the initial full dataset (inclusive of outliers) with the caret package:
standardizing the data help us to clearly capture how each variable equally contributes to the analysis.
If we don't standardize, the models will be dominated by the variables that use a larger scale, adversely affecting the performance. 
In this project, standardization is achieved with the use of the “preProcess” function: the first line of the code loads the pre-processes the data, only for the columns I am interested in.
The second line performs the normalization, while the third command prints the summary of the standardized variable: the output shows “normdata” a new dataset with all numerical variables standardized with a mean value of zero. Now that the dataset is standardized, we can proceed with building the models.

Although the assumptions to apply linear regression to the full dataset are not fully met, we will still create a linear regression model to be used as naive method for comparison to the other models.
Residuals are checked with the use of an histogram; the fitted values and residuals plot are calculated next, to check the assumption of homoscedasticity (which is when the size of the error doesn’t change significantly across the values of the independent variable).
Finally, the linear model is plotted with the data.

As we see and predicted, the model does not fit well, so we will next try and build a polynomial regression model instead: “poly_model”.
The summary of this model shows the “Residuals” which give information of how well the model fits around the real data. Next, is the “Coefficients” table where the first row, labelled “Intercept”, gives us the y-intercept value of the regression equation, in this example 0.0008206744.
The next row in the Coefficients table is the first independent variable Beds: each following row describes the estimated effect of the independent variable on the dependent variable Price.
Now, looking at each column, we have the “Estimate” column (known as the regression coefficient or R^2 value) is the estimated effect for Beds: for every one unit increase in Beds there is a corresponding decrease in -0.0196050590 unit reported in the Price. 
The “Std. Error” column displays the standard error of the estimate which tells us how much variation there is in our estimate of the relationship between Beds and Price.
The “t value” column displays the test statistic from a two-sided t-test: the larger the test statistic, the less likely it is that our results occurred by chance. So, higher the t-value, the better.
Lastly, the “Pr(>| t |)” column shows the p-value which tells us how likely we are to see the estimated effect of Beds on Price if the null hypothesis of no effect were true.
If we look below, we’ll see also the last three lines of the model summary, which are statistics about the model as a whole. The most important value to notice here is the p-value of the model.
Here it is significant (2.401e-16 in decimal form is 0.0000000000000002401), which means that this model is a good fit for the observed data.
Both p-values (the one shown at the bottom of the last line and the p-value of individual predictor variables in the “Pr(>| t |)” column), are very important because we can consider a model to be statistically significant only when both these values are less that the pre-determined statistical significance level, which is 0.05. This significance is visually interpreted by the stars at the end of each row of the table: the more the stars beside the variable’s p-value, the more significant the variable.
If useful to specify that whenever there is a p-value, there is a null and alternative hypothesis associated with it. 
The Null Hypothesis is that the coefficients associated with the variables is equal to zero (so there is no relationship between the independent variable in question and the dependent variable). The alternate hypothesis is that the coefficients are not equal to zero (so there exists a relationship between the independent variable in question and the dependent variable).
Pr(>|t|) is therefore the probability that you get a t-value as high or higher than the observed value, when the Null Hypothesis is true: if the Pr(>|t|) is low, the coefficients are significant (significantly different from zero); on the other hand, if the Pr(>|t|) is high, the coefficients are not significant (non-significantly different from zero).
Overall, the more significant independent variables of this model are, in order: Baths, Building Area and Latitud; both the general p-value and the p-value associated to each variable are below the 0.05 threshold, so we can conclude that our model is statistically significant.

With our polynomial mode “poly_model”, we go ahead and check the errors residuals with the diagnostic plots which show the unexplained variance (residuals) across the range of the observed data. Each plot gives a specific piece of information about the model fit with the red line representing the mean of the residuals. The line should be horizontal and cantered on zero (or on one, in the scale-location plot) meaning that there are no large outliers that would cause bias in the model.
The normal Q-Q plot, shows a regression between the theoretical residuals of a perfectly-homoscedastic model and the actual residuals of the model: the closer to a slope of 1, the better. 
This plot is there to indicate if the distribution is normal, if It is normal (means not biased) there would be a linear structure.
In the scale-location plot (similar to the previous plot) the red line must be horizontal as possible and the spots should be distributed around it, without a particular structure to confirm an unbiased estimate of the data.
Finally, the residuals vs leverage graph should show no particularly pronounced pattern on the data and the line should be straight to confirm again non bias.

A comparison between our naïve linear model and this polynomial model (“poly_model”), is then performed using the regression Technique ANOVA ‘F’ Test where the error of all the implemented models is analysed. As shown, the “RSS” (residuals sum of spares) score of the polynomial model is less concerning than the linear mode, which means that the polynomial model is the best one.
The significance stars are also shown for the polynomial model for the ‘F’ score which can be translated as the significance probability associated with the statistic and measures how much variation in the dependent variable can be accounted for by the model.
So we concluded that our polynomial regression models have the better solution for this dataset. 
Finally, the curve model of this polynomial model is plotted with the data with a ggplot.
From the plot we can see that the model does not entirely fit the dataset yet, but we will now try and optimize the model. To improve the polynomial model, backward elimination has been used to remove the predictor with the largest p-value (> 0.05). In this case, we will remove variables "Beds" & "Longitude" first, then fit the model again, into another polynomial model called “fit”.
The summary of this model shows that it can further be improved, so we do the improvement all over again by removing all the additional predictors with the largest p-value, over 0.05. In this case, we will remove also variable "Latitude", then fit the model again into another polynomial model called “fit_ok”.
Both models’ residuals errors will be analysed with an ANOVA ‘F’ Test performed to check which is the best performing model.
Another ANOVA check, between the previously created poly_model and the latest improved models (fit and fit_ok) is performed, showing that the polynomial fitted model with independent variables Baths, Building Area and Latitude (model fit) is the slightly better performing one.

Now that we have a general understanding of the performance of the full datasets, the project will now focus on studying the performance of the models on the earlier created dataset without outliers (“new_df”) which we recall includes only house prices below 500,000$ and is made up of 8,458 observations.
Like earlier, the standardization of this dataset is done using “preProcess” and the outcome is “normdata_noout” our standardized dataset without outliers with all numerical variables standardized with a mean value of zero.
Given the fact that we still want to have a naïve linear model for comparison, both linear and polynomial models’ regressions are created, with all variables; backward elimination is then applied to both models to remove the predictor with the largest p-value, over 0.05.
The backward elimination creates 2 additional models (one linear “linear_fit_noot” and one polynomial “poly_fit_noout”).
The newly created and improved models are then confronted with an ANOVA ‘F’ Test between each other to locate the best performing oneS (which appear to be both polynomial models “poly_model_noout” with all of the independent variables, and “poly_fit_noout” with independent variables Baths, Latitude, Longitude only).
Finally, these best performing polynomial models are plotted against the dataset, using again ggplot.

The next part of the project will see some predictions made on all of the models of the full dataset first and on all of the models of the dataset without outliers, secondly.
The best performing polynomial models in both datasets (models fit & poly_model for the full dataset - and poly_fit_noout & poly_model_noout models for the cleaned dataset) will be checked first and for each of those, confidence internals and plots to better investigate the predictions will be shown.
The plots will show the fitted model versus the residuals, but also the predicted values versus the actual values of the models.
A final and more comprehensive plot will visualize prediction intervals with the dataset, together with the models’ regression line.
R^2 (the value of R-squared) for the prediction model on the dataset, will be calculated for every model to evaluate the goodness of fit of the model itself.
As mentioned earlier, what R-Squared tells us is the proportion of variation in the dependent (response) variable that has been explained by the model. This is calculated as 1 minus the sum of squared errors given by SSE divided for the SST, which is the sum of squared total.
Higher R^2 is better, with value 1 being the best.

According to the R-squared results, the project will next analyse the 2 best performing models, from the dataset without outliers (again models: poly_fit_noout & poly_model_noout) firstly, and from the full dataset (again models: fit & poly_model) secondly.
We will be dividing the data sets into two parts: training and validation set and we will also specify the percentages of the data in these two divisions.
Our cleaned dataset will be splitted into training and testing set with a 70/30 split, with 5920 training examples and 2538 testing (or validation) examples, setting the seed at 1 to so that we get the same partitions when re-running the code.
We are then calculating the regression model with the train data and predicting the value of the house prices with it. We will test the first few rows of the header to see the prediction results; we will then be using the validation data to validate the model and measure the accuracy of it by also confronting the R-squared values of the traindata versus the validdata.
The same steps will be repeated for each of the best performing models within the full dataset, which will be splitted again, into training and testing set with a 70/30 split, with 6994 training examples and 2538 testing (or validation) examples, still setting the seed at 1 .
As earlier, for each best performing model, the regression model with the train data is calculated, predictions of the value of the house prices are made with it.
The validation data is then used to validate the model and to check how well does the model work on the validation data; finally, R squared values are calculated for both the traindata and the validdata.
Due to the results of the regression models being quite underwhelming, an additional analysis will be done using the Support Vector Regression (SVM analysis), on the dataset without outliers only, being the better performing dataset.
Support Vector Machine is a supervised learning method and it can be used for regression and classification problems. The “e1071” package provides “svm” function to build support vector machines model to apply for regression problem in R. 
As earlier, we'll prepare the full data by splitting it into the train and valid data, partitioned at 70/30.
We will define the svm model with default parameters and fit it with the traindata, with the following script:

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/398f939f-7f82-44f1-ab77-6491ab9734b2)

Next, we'll predict the validdata and plot the results to compare, visually.
Finally, we'll check the prediction accuracy with the following metrics:
MSE = mean squared error 
MAE = mean absolute error
RMSE = Root Mean Square Error
By calculating error rates MSE, MAE and the RMSE, we can find out the prediction accuracy of the model (the errors should be the smaller possible). The R-squared, for model is also being calculated (as we will see, being the best R-squared registered yet).

# 4.	Results 
The full dataset’s assumptions to apply linear regression are shown by the results below: the matrix scatter plot shows the relationship between independent variables Beds, Baths, Building Area, Latitude and Longitude.
There isn’t a clear linear relationship between any of the variables.
However, it appears that Baths has the strongest relationship with Beds, thanks to a rounded Pearson’s correlation coefficient of (r = 0.32).
There is also some relationship between Building area and Baths; but also, between Baths and Longitude/Latitude and with a negative correlation of -0.12 between Latitude and Longitude as well.

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/3a91f0b0-cdd1-4f3b-b223-145918219a53)

The dependent variable Price, does not follow a normal distribution: it appears to be skewed to the right as the histogram below show.

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/95c08ae9-51e4-4cad-b0c0-bc6700740085)

The assumptions to apply linear regression are not met: the full dataset is not the best dataset suitable for linear regression.

The same assumptions are then checked to the dataset without outliers: the matrix scatter plot below, shows the relationship between all the independent variables. Again, there isn’t a clear linear relationship between any of the variables.
However, it appears that Beds has the strongest relationship with Baths (r = 0.20) with also a negative correlation of -0.17 between Longitude and Latitude.
There is also some relationship between Building area and Baths and between Baths and  Longitude/Latitude, with a negative correlation of -0.12 between Latitude and Longitude.

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/463a80ff-3791-4129-99ca-c8ee1fec7f8a)

The dependent variable Price, still does not follow a perfectly normal distribution: it appears to be slightly skewed to the right, as the histogram below shows:

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/91c962c7-ac9e-49db-b2ad-d398055203e2)

Finally, it is good to see how the prices are highlighted in this plot with Latitude and Longitude in the x and y axis: the majority of the cheaper houses are located at < -100 in Longitude.

We conclude that, again, the assumptions to apply linear regression are not fully met: the dataset without outliers may not be the best dataset suitable for linear regression.
As a consequence, we now then create a polynomial model instead.
The summary of the “poly_model” is the below:

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/bc43c80d-b606-4418-8364-9cc6ead2bf8a)

The summary shows that the p-value of the model is significant (2.401e-16), which means that this model is a good fit for the observed data.
The statistically significant variables (Baths, Building Area and - Latitude in minor part-) let us reject the null hypothesis that the co-efficient of the predictor is zero, so we conclude that our model is indeed statistically significant and that a polynomial regression model is the better solution for this dataset. 
Here below a plot which shows the fitting of the poly_model model to the full dataset.

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/c62fc51d-9e5d-45f9-9b8c-6578d8ab066f)

The residuals check of this model (shown in the next page) present in the Residuals vs Fitted plot with the red line being fitting well on the straight line, meaning that the model may be unbiased.
In the Normal Q-Q plot some values appear to be okay, others however aren't: the graph has a S shape instead of a line, so we conclude that the model is not a perfectly unbiased estimate of the data.
The Scale-Location plot, appears to have a slight structure as the line is not really horizontal and the data is mostly concentrated on the right side, the model could indeed be biased.
Finally, the Residuals vs Leverage plot shows a particularly pronounced pattern on the data and the line is non straight, confirming again there could be indeed bias.
From these diagnostic plots we can say that the model does not fit the assumption of homoscedasticity and may be biased.

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/8913a497-be48-4cc7-8725-a2bfa9c7ac72)
![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/bc68bcc4-a38a-44f2-8988-7b85d7403468)


![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/5b93b59d-9d07-4ece-959e-c9d2f72643fd)
![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/8aa73079-d53f-4138-ba7b-866c5aab1ec2)

Following these results, it now makes even more sense to analyse the fitness of the models to the dataset without outliers.
The best performing polynomial model for the dataset without outliers appears to be “poly_fit_noout” as the summary below shows:

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/1827677f-2618-4b0e-b8c9-5dbf275d37c0)

The p-value of this model is significant (2.401e-16) which means that this model is a good fit for the observed data.
All of the variables appear to be statistically significant, so we can reject the null hypothesis and we conclude that our model is statistically significant (even better than the previous model calculated with the full dataset).

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/c5af9d31-1b2c-4f5e-8b7a-1b1a8bebea12)
![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/f1b1b6d9-7d26-4045-a072-69bc4e9749c3)

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/86e990b9-b756-46a6-b7a0-3c134bb26054)
![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/6efe1105-fcae-4ad8-8d4d-208851d6e91d)

The residuals check of this model appear to be still a bit biased: in the Residuals vs Fitted plot, the red line does not fit too well on the straight line, meaning that the model may be biased.
The Normal Q-Q plot, still shows a S shape; the Scale-Location plot, appears to have a slight structure as the line is not particularly horizontal.
Finally, the Residuals vs Leverage plot doesn’t show a particularly pronounced pattern on the data and the line is almost straight, we conclude here that here may be not bias.
The residuals check is slightly better in this model, still not enough to fit the assumption of homoscedasticity.

Now, looking at the predictions of the 2 best performing models within the full dataset (models fit & poly_model) the graphs below depict the prediction intervals highlighted in orange with the blue line being the model regression line:

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/e076c7c6-d1c1-4bc5-8e9d-1753367d077f)

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/b24dbd36-e158-4253-bd5f-13e39115e75e)

The R-squared values calculated for the models are 0.009645532 for the poly_model and 0.008576954 for fit model, with the poly_model (second graph) performing slightly better than the first.

The predictions of the 2 best performing models in the in the dataset without outliers (models poly_fit_noout & poly_model_noout) are shown in the graphs below which again depict the prediction intervals in orange with the blue line being the model’s regression line:

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/ae531f2e-e500-4bdb-9296-f5a0ba40c376)

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/72680078-58d1-4521-8cda-6245c4fdab0f)

The R-squared value calculated for these models are 0.1262146 for the poly_model_noout and 0.1269805 for poly_fit_noout, we conclude that the best model is the one showed in the second graph.
poly_model_noout is also the best performing polynomial model within the whole project.
A summary of all the relevant results of all the models is shown in the summary table below:

<img width="391" alt="image" src="https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/7845e94b-d170-44e1-801b-f11eecbecd05">
From the table we can see that, generally, the dataset without outliers performs better than the full dataset. As we have seen, polynomial regressions outperform linear regressions, producing better statistics and results.

Lastly, the SVM model table below shows a summary of all of the results from this model which has been applied to the dataset without outliers only: with an R-squared of 0.2500602, SVM appears to be the best performing model of the whole project.

<img width="407" alt="image" src="https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/65f0f2b6-1ad2-4369-95c0-91097d533fc3">

Interesting to view, is also the graph below which shows the values of the Price in the validate as red dots against the blue line being the predicted valid data within the SVM model.

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/0509feee-28c3-4d3c-95a4-95fb5bd64c01)

# 5.	Discussion & Conclusions
Linear, polynomial regression and SVM analysis were carried out to investigate the relationship between the dependent variable House Price and the independent variables Beds, Baths, Building area, Latitude and Longitude. 
Due to the assumptions to perform linear regression being not met, polynomial regression analysis was undertaken instead (the linear regression model was still calculated and considered as a naïve method for comparison to the other models).
The summary of the polynomial model (poly_model) shows that the p-value of the model is significant (a very small 2.401e-16); there is significant relationship between Price and Baths (t = 7.313, p = 2.81e-13), Price and Building Area (t = 2.460, p = 0.01392) and in a smaller amount, between Price and Latitude (t= 2.086, p = 0.03698). The R-squared value on the validation dataset is however pretty small, 0.08662633: only 8,66% of the variation in Price can be explained by this model.

A check of the residuals of this model, shows that the model itself, does not fit the assumption of homoscedasticity and therefore is biased.
To try and overcome the bias and to double check the dataset behaviour, the dataset has been reduced to exclude outliers: only house prices below price 500,000$ have been considered in the new_df dataset.
Reducing the dataset seems to create slightly better results: the summary of the fitted polynomial model (model poly_fit_noout) shows that the p-value of the model is significant (a very small < 2.2e-16); there is significant relationship between Price and Baths (t = 32.782, p = < 2e-16), Price and Latitude (t = 9.984, p = < 2e-16) and Price and Longitude (t= -5.200, p = 2.03e-07). The R-squared value on the prediction model is still small but improved: 12,70% of the variation in Price can be explained by this model.
According to the summary table above, the dataset without outliers performs better for every model that has been created, with the polynomial models being the ones with the overall better results.
As the general results with regression analysis are quite low, a Support Vector Machine analysis is made to double check if the dataset without outliers can be further improved.
The overall performance of the dataset with this model is indeed improved: with an R-squared of 0.2500602, SVM appears to be the best performing model of the whole project because 25,00% of the variation in Price can be explained by this model.
The dataset appears to be quite difficult to be analysed with big values influencing the residuals errors checks of the models.
We have seen that the regression functions are substantially changed by the removal of outliers in the full dataset, however these changes are not producing significantly better results.
Ignoring valid data values by removing the outliers, may not be the best solution because while removing these outliers slightly improves the models, it does not necessarily lead to a significantly greater R-square value for the new fitted model, or to a smaller P value for the F test of overall fit.
Furthermore, valuable and significant information is lost by doing the resizing of the dataset, creating implicit biases to the overall performance.
Some feature directions could be analysing the dataset in terms of particular geographical area or particular state: by analysing only selected subset parts of the full dataset, we may be able to produce better results and discover hidden patterns or relationships.
Additionally, creating a more complex model, better fitted to the dataset with the use of classification instead of regression (we could use for example Naive Bayes) may help to improve the fit of the model to the full dataset.





























































