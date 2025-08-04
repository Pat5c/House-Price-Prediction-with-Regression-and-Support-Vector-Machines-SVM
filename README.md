# House Price Prediction with Regression and Support Vector Machines (SVM)

<img width="573" height="571" alt="image" src="https://github.com/user-attachments/assets/0450a127-a36f-4fce-91ca-3c0ab7b36178" />

# Table of Contents

1.	Abstract ……………………………………………………………………………………………………………………3

2.	Introduction …………………………………………………………………………………………………………4

3.	Methods ………………………………………………………………………………………………………………………6

4.	Results ………………………………………………………………………………………………………………………13

5.	Discussion & Conclusions …………………………………………………………………………24

6.	Bibliography …………………………………………………………………………………………………………26

7.	Appendices ………………………………………………………………………………………………………………27

# 1.	Abstract 

This project focuses on predicting U.S. house prices using regression techniques. Key influencing factors such as bedrooms, bathrooms, building area, location (longitude and latitude) are analyzed to determine their impact on price prediction.
Multiple models are evaluated, including Linear Regression, Polynomial Regression, and Support Vector Machines (SVM). The study explores how data quality—particularly the presence or removal of outliers—affects model performance.

Findings show that:
- Linear models perform poorly on raw data (R² ≈ 0.0078),
- Polynomial Regression and SVM models perform better on cleaned data, achieving R² scores of 0.1280 and 0.2500 respectively.

The project highlights the importance of feature selection, data preprocessing, and model choice in achieving better predictive accuracy.

# 2.	Introduction 

Buying a house is asily the biggest and most important financial investment individuals can do during their life time, and accurately predicting house prices is critical for both buyers and real estate agents. This project uses a comprehensive U.S. housing dataset (9,992 records from June 2021) to explore data-driven price prediction using regression techniques.
After initially analysing the suitability of the dataset for linear regression, by double checking the assumptions for regression, the steps taken are:

- Data cleaning (removing outliers with prices over $500,000).
- Normalization of numeric features for better model performance: as it will be shown in the project, the polynomial regression model is better fitted on both datasets, but   we will still build the linear regression model in order to make comparisons and analyse the suitability of the model to the full dataset and to the cleaned dataset   (without outliers).
- Modeling using both Linear and Polynomial Regression, and also:
- Support Vector Machines (SVM) for regression, appropriate for this continuous prediction task, and it will ultimately yield the best performance.

Feature selection is refined using backward elimination (based on p-values) and ANOVA with F-test will be implemented to help us analyse the error of all the implemented models. Each model’s effectiveness is compared using R-squared values, helping determine the most accurate and best-fitted model: higher R-squared means better result, with 1 being the best. Lastly, the dataset is split into training and validation sets to evaluate model performance more reliably.

# 3.	Methods

The dataset, sourced from ZenRows (2021), contains 10,000 records of residential properties across the United States and includes 47 features such as price, location, and house characteristics.

#### Key Processing Steps:
- Reduced to 11 core features: ID, Price, Address, City, Zipcode, Bedrooms, Bathrooms, Building Area, Latitude, and Longitude.
- Removed rows with missing price values, resulting in 9,992 clean records.

#### Handled missing data:
- Replaced "Not provided" to the column Address
- Replaced "0" to column Zipcode.
- Filled missing numerical values in the other columns (Beds, Baths, Building area, Latitude and longitude) with respective column means.

The cleaned dataset is structured and ready for regression modeling, with appropriate data types and no critical missing values.

## The modeling process began by:
1. Installing necessary libraries and importing the cleaned dataset (us-cities-real-estate-sample-zenrows.csv),
2. Checking linear regression assumptions:
  - Independence: Checked via Pearson correlation.
  - Normality: Assessed using histograms of the target variable (Price).
  - Linearity: Visualized with scatter plots comparing Price to Beds, Baths, Building Area, Latitude, and Longitude.
  - Multicollinearity: Assessed with scatterplot matrix.

Due to unmet assumptions on the full dataset, a reduced dataset was created by filtering out houses priced above $500,000 (8,458 records). Both datasets were then standardized using the caret package to ensure all features contribute equally to modeling.

#### Model Building & Evaluation:
1. Linear Regression (baseline)
  - Built on both full and reduced datasets.
  - Showed poor model fit (low R², non-constant variance of residuals).
  - Served as a naive benchmark.

2. Polynomial Regression
  - Showed better fit than linear regression.
  - Evaluated with:
    - Coefficient significance (p-values)
    - Diagnostic plots (Q-Q plot, residuals vs fitted, scale-location, leverage)
  - ANOVA ‘F’ test confirmed better performance over the linear model.
  - Final model (after backward elimination of variables with p > 0.05) retained Baths, Building Area, and Latitude as significant predictors:
  - Polynomial model (fit) emerged as the best-performing regression model.

This process highlighted that while a simple linear model was insufficient, a well-tuned polynomial regression significantly improved predictive performance and interpretability.

#### Outlier-Free Dataset Modeling & Prediction
The next phase focuses on the cleaned dataset without outliers (new_df, 8,458 records), where:
- The data is standardized again (normdata_noout).
- Both linear and polynomial models are built.
- Backward elimination is applied to refine the models, resulting in:
    - linear_fit_noout (reduced linear model)
    - poly_fit_noout (reduced polynomial model with key predictors: Baths, Latitude, Longitude)

## ✅ Best Models Identified
Using ANOVA F-tests, the best-performing models are:

- On outlier-free data: poly_model_noout and poly_fit_noout
- On full dataset: poly_model and fit

These are visualized with ggplot to compare predictions and residuals.
All selected models are then used to make predictions, with:
- Residual plots and prediction vs actual plots
- Prediction intervals overlaid on data
- R² values calculated to measure model fit


## 🧪 Train/Test Split & Model Validation
Each dataset (full and outlier-free) is split:
- 70% Training / 30% Validation, with set.seed(1)
  Cleaned dataset: 5,920 training / 2,538 testing
  Full dataset: 6,994 training / 3,006 testing

For each best model:
- Trained on the training set
- Used to predict validation set outcomes
- R² scores computed for both training and validation sets to assess model generalization

## 🤖 Support Vector Regression (SVM)
To improve performance, Support Vector Machines (SVM) are applied to the outlier-free dataset, which already performed best with regression models.

Process:
- Data split into 70/30 train/test sets
- We will define the svm model with default parameters and fit it with the traindata, with the following script:
![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/398f939f-7f82-44f1-ab77-6491ab9734b2)
- SVM model built using e1071::svm() with default parameters
- Model trained and used to predict house prices

## 📊 Evaluation Metrics:
MSE (Mean Squared Error)
MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
R² score

✅ SVM achieved the highest R² value, making it the best-performing model overall in this project.

# 4.	Results 

#### Linear Assumptions & Dataset Suitability
✅ Full Dataset:
- No strong linear relationships among predictors (e.g., Baths & Beds have the highest correlation, r = 0.32).
  ![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/3a91f0b0-cdd1-4f3b-b223-145918219a53)
- Price (the dependent variable) is not normally distributed — it's right-skewed, violating linear regression assumptions.
![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/95c08ae9-51e4-4cad-b0c0-bc6700740085)
- Diagnostic plots show violations of homoscedasticity and potential bias.


✅ Dataset Without Outliers:
- Similar correlation patterns (e.g., Beds & Baths r = 0.20).
- Slight improvement in distribution (still right-skewed).
  ![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/463a80ff-3791-4129-99ca-c8ee1fec7f8a)
- Still fails linear regression assumptions, but less biased than the full dataset.

🧮 Polynomial Regression Models
🔧 Full Dataset:
- poly_model shows significant predictors: Baths, Building Area, Latitude.
  ![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/bc43c80d-b606-4418-8364-9cc6ead2bf8a)
- p-value: 2.401e-16 → model is statistically significant.
- Diagnostic plots reveal bias and non-constant variance.
- R² ≈ 0.0096 → low explanatory power.
  
Here below a plot which shows the fitting of the poly_model model to the full dataset:
  ![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/c62fc51d-9e5d-45f9-9b8c-6578d8ab066f)


🔧 Dataset Without Outliers:
- poly_fit_noout has all significant predictors and better diagnostics.
- p-value: 2.401e-16, R² ≈ 0.127
- Less bias, better homoscedasticity, and superior performance than the full dataset model.

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/8913a497-be48-4cc7-8725-a2bfa9c7ac72)
The residuals check of this model (shown in the next page) present in the Residuals vs Fitted plot with the red line being fitting well on the straight line, meaning that the model may be unbiased.

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/bc68bcc4-a38a-44f2-8988-7b85d7403468)
In the Normal Q-Q plot some values appear to be okay, others however aren't: the graph has a S shape instead of a line, so we conclude that the model is not a perfectly unbiased estimate of the data.

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/5b93b59d-9d07-4ece-959e-c9d2f72643fd)
The Scale-Location plot, appears to have a slight structure as the line is not really horizontal and the data is mostly concentrated on the right side, the model could indeed be biased.

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/8aa73079-d53f-4138-ba7b-866c5aab1ec2)
Finally, the Residuals vs Leverage plot shows a particularly pronounced pattern on the data and the line is non straight, confirming again there could be indeed bias.
From these diagnostic plots we can say that the model does not fit the assumption of homoscedasticity and may be biased.
  
- Identified as the best polynomial regression model for the dataset without outliers as the summary below shows:

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/1827677f-2618-4b0e-b8c9-5dbf275d37c0)

The p-value of this model is significant (2.401e-16) which means that this model is a good fit for the observed data. All of the variables appear to be statistically significant, so we can reject the null hypothesis and we conclude that our model is statistically significant (even better than the previous model calculated with the full dataset).

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/c5af9d31-1b2c-4f5e-8b7a-1b1a8bebea12)
![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/f1b1b6d9-7d26-4045-a072-69bc4e9749c3)

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/86e990b9-b756-46a6-b7a0-3c134bb26054)
![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/6efe1105-fcae-4ad8-8d4d-208851d6e91d)

📈 Model Comparison – R² Scores
Model	Dataset	R² Score
fit (Linear)	Full	0.0086
poly_model	Full	0.0096
poly_model_noout	Without Outliers	0.1262
poly_fit_noout	Without Outliers	0.1270
SVM	Without Outliers	0.2501 ✅

Now, looking at the predictions of the 2 best performing models within the full dataset (models fit & poly_model) the graphs below depict the prediction intervals highlighted in orange with the blue line being the model regression line:

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/e076c7c6-d1c1-4bc5-8e9d-1753367d077f)

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/b24dbd36-e158-4253-bd5f-13e39115e75e)

The R-squared values calculated for the models are 0.009645532 for the poly_model and 0.008576954 for fit model, with the poly_model (second graph) performing slightly better than the first.

The predictions of the 2 best performing models in the in the dataset without outliers (models poly_fit_noout & poly_model_noout) are shown in the graphs below which again depict the prediction intervals in orange with the blue line being the model’s regression line:

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/ae531f2e-e500-4bdb-9296-f5a0ba40c376)

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/72680078-58d1-4521-8cda-6245c4fdab0f)

The R-squared value calculated for these models are 0.1262146 for the poly_model_noout and 0.1269805 for poly_fit_noout, we conclude that the best model is the one showed in the second graph.
poly_model_noout is also the best performing polynomial model within the whole project.

🤖 SVM (Support Vector Machine) Model
Applied only to outlier-free dataset

Best performance across all models:
- R² = 0.2501
- Low MSE and RMSE

Visualizations show SVM predictions closely tracking actual house prices in the validation set.

Lastly, the SVM model table below shows a summary of all of the results from this model which has been applied to the dataset without outliers only: with an R-squared of 0.2500602, SVM appears to be the best performing model of the whole project.

<img width="407" alt="image" src="https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/65f0f2b6-1ad2-4369-95c0-91097d533fc3">

Interesting to view, is also the graph below which shows the values of the Price in the validate as red dots against the blue line being the predicted valid data within the SVM model.

![image](https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/0509feee-28c3-4d3c-95a4-95fb5bd64c01)

✅ Key Takeaways
- Removing outliers significantly improved model performance.
- Polynomial models outperformed linear models.
- SVM model provided the best overall accuracy and generalization.

For real-world applications, non-linear models like SVM are preferable for housing price prediction on this dataset.

# 5.	Discussion & Conclusions

📊 Results Summary
Three models—Linear Regression, Polynomial Regression, and Support Vector Machine (SVM)—were tested to predict house prices based on key features: Beds, Baths, Building Area, Latitude, and Longitude.

🔹 Linear vs Polynomial Regression
- The dataset violated key assumptions for applying linear regression (e.g., no clear linearity, non-normality, heteroscedasticity), so polynomial regression was explored instead.

- Polynomial regression (poly_model) showed statistically significant relationships:
Baths (p ≈ 2.81e-13), Building Area (p ≈ 0.014), Latitude (p ≈ 0.037).

- However, it explained only 8.66% of the variation in house prices (R² = 0.0866).
- Residual diagnostics revealed bias and lack of homoscedasticity.

🔹 Impact of Removing Outliers
- A refined dataset (Price < $500,000) improved performance:
- poly_fit_noout had stronger significance across all variables and a better R² of 12.7%.
- Still, the model was not ideal—residuals indicated persistent bias.

🔹 Best Model: Support Vector Machine
Applying SVM to the cleaned dataset yielded the best performance:
- R² = 0.2501, meaning the model explained 25% of price variation.
- This was the highest performing model of the project.

🔍 Key Insights
- Removing outliers improves models slightly, but at the cost of losing potentially valuable information and introducing bias.
  
A better approach may be:

- Analyzing specific geographic subsets (e.g. state-level modeling).
- Exploring classification models like Naive Bayes for categorizing price bands.

A summary of all the relevant results of all the models is shown in the summary table below:

<img width="391" alt="image" src="https://github.com/Pat5c/House-Price-Prediction-with-Regression-and-Support-Vector-Machines-SVM/assets/124057584/7845e94b-d170-44e1-801b-f11eecbecd05">
From the table we can see that, generally, the dataset without outliers performs better than the full dataset. As we have seen, polynomial regressions outperform linear regressions, producing better statistics and results.






























































