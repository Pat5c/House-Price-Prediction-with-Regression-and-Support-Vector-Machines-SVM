---------##############CW2####################--------------------------
#Loading all required packages, line by line
install.packages("tidyverse")
library(tidyverse)
install.packages("ggplot2")
library(ggplot2)
install.packages("caret")
library(caret) 
library(dplyr)
install.packages("e1071", dependencies = TRUE)
library(e1071)
install.packages("Metrics")   #Install & load Metrics package
library("Metrics")

#Uploading the Dataset: please replace ""C:/Users/patie/OneDrive/Documents/" with your OWN DIRECTORY
USRealEstate_data <- read.csv("C:/Users/patie/OneDrive/Documents/us-cities-real-estate-sample-zenrows.csv")
View(USRealEstate_data)
str(USRealEstate_data)
summary(USRealEstate_data)

#Setting the correct data types
col.selection <- c("Price","Beds","Building.Area")
USRealEstate_data[col.selection] <- sapply(USRealEstate_data[col.selection], as.numeric)
str(USRealEstate_data)

#Checking all the assumptions to apply linear regression to the Dataset:

#ASSUMPTION 1: INDEPENDENCE OF OBSERVATIONS, NO AUTOCORRELATION
#Producing a table of Pearson's correlation coefficients rounded to two decimal places:
round(cor(cbind(USRealEstate_data$Beds, USRealEstate_data$Baths, USRealEstate_data$Building.Area, USRealEstate_data$Latitude, USRealEstate_data$Longitude)),2)

#ASSUMPTION 2: NORMALITY, to check whether the dependent variable follows a normal distribution
options("scipen"=5) #Showing scientific number in full numbers
hist(USRealEstate_data$Price)
hist(USRealEstate_data$Price, xlim = c(0, 10000000),
     ylim = c(0, 500), breaks = 50000,
     xlab='Price',
     ylab='Frequency',
     main='Histogram of Price') #Zooming in

#ASSUMPTION 3: LINEARITY
#Price & Beds
options("scipen"=5) #Showing scientific number in full numbers
a <- ggplot(USRealEstate_data, aes(x=Beds, y=Price)) + geom_point()
a
b <- a + ylim(0, 10000000) + xlim(0, 50) #zooming in to 
b

#Price & Baths
options("scipen"=5)
c <- ggplot(USRealEstate_data, aes(x=Baths, y=Price)) + geom_point()
c
d <- c + ylim(0, 10000000) + xlim(0, 20) #zooming in to 
d

#Price & Building.Area
e <- ggplot(USRealEstate_data, aes(x=Building.Area, y=Price)) + geom_point()
e
f <- e + ylim(0, 10000000) + xlim(0, 100000) #zooming in to 
f
g <- f + ylim(0, 1000000) + xlim(0, 25000) #zooming in to 
g

#Price & Latitude and Longitude
h <- ggplot(USRealEstate_data, aes(x = Latitude, y = Longitude)) + geom_point(aes(colour = Price)) + xlab("Latitude") + ylab("Longitude") + theme(axis.line = element_line(colour = "black", size = 0.24))
h

# Relationship between Beds, Baths, Building.Area, Latitude and Longitude"
pairs(USRealEstate_data[,c(7:11)], col= "blue", pch=18, main= "Relationship between Beds, Baths, Building.Area, Latitude and Longitude")

##############---------------##################
#REMOVING OUTLIERS AND TAKING INTO CONSIDERATION OLY PRICES < 500000

#Creating this new_df without outliers
new_df <- subset(USRealEstate_data, USRealEstate_data$Price < 500000)
View(new_df)
str(new_df)

#Checking again all of the assumptions to apply linear regression to the model:

#ASSUMPTION 1:INDEPENDENCE OF OBSERVATIONS, NO AUTOCORRELATION
#Producing a table of Pearson's correlation coefficients rounded to two decimal places:
round(cor(cbind(new_df$Beds, new_df$Baths, new_df$Building.Area, new_df$Latitude, new_df$Longitude)),2)

#ASSUMPTION 2: NORMALITY to check whether the dependent variable follows a normal distribution
hist(new_df$Price, xlab='Price',
     ylab='Frequency',
     main='Histogram of Price') #Zooming in)

#ASSUMPTION 3: LINEARITY
#Price & Beds
options("scipen"=5) #Showing scientific number in full numbers
ggplot(new_df, aes(x=Beds, y=Price)) + geom_point()

#Price & Baths
options("scipen"=5)
ggplot(new_df, aes(x=Baths, y=Price)) + geom_point()

#Price & Building.Area
ggplot(new_df, aes(x=Building.Area, y=Price)) + geom_point()

#Price & Latitude and Longitude
ggplot(new_df, aes(x = Latitude, y = Longitude)) + 
  geom_point(aes(colour = Price)) + 
  xlab("Latitude") + 
  ylab("Longitude") + 
  labs(title="Price distribution in Latitude and Longitude") +
  theme(axis.line = element_line(colour = "black", size = 0.24))

# Relationship between Beds, Baths, Building.Area, Latitude and Longitude"
pairs(new_df[,c(7:11)], col= "blue", pch=18, main= "Relationship between Beds, Baths, Building.Area, Latitude and Longitude")

##########################--------------##################
# STANDARDIZING THE DATASET

#Proceeding to standardize the initial complete data (including outliers) with library caret
preprocdata <- preProcess(USRealEstate_data[,c(2,7:11)], method=c("center", "scale"))
normdata <-predict(preprocdata, USRealEstate_data[,c(2,7:11)])
summary(normdata)
View(normdata)

#Residual check for normalized full dataset:
plot(normdata)

################---------------------------##############################
#From the checks above, We conclude that the linear model is not the best model as not all assumptions are met. However, we will still build it to check and compare the results and fit the model to both datasets (with outliers now and without outliers later)

#Building a multiple linear model to calculate the effect of the response variable "Price" on the explanatory variables: Beds, Baths, Building.Area, Latitude and Longitude:
linear_model <- lm(normdata$Price ~ normdata$Beds + normdata$Baths + normdata$Building.Area + normdata$Latitude + normdata$Longitude)
summary(linear_model)

#To check the residuals, I am producing an histogram of standardised residuals
hist(resid(linear_model), main='Histogram of residuals',xlab='Standardised
Residuals',ylab='Frequency')
hist(resid(linear_model), main='Histogram of residuals', xlab='Standardised Residuals', ylab="Frequency", xlim = c(0, 20), ylim = c(0, 20), breaks = 50) #Zooming in

#The fitted values and residuals plot are calculated next to check the assumption of homoscedasticity.
plot(linear_model)

#Plotting the model with the data
ggplot(data = normdata, aes(x = Beds + Baths + Building.Area + Latitude + Longitude, y = Price)) + geom_point(alpha = 1/10) + geom_smooth(method = 'lm', formula = y ~ x)
#Zooming in
ggplot(data = normdata, aes(x = Beds + Baths + Building.Area + Latitude + Longitude, y = Price)) + ylim(0, 5) + xlim(-10, 10) + geom_point(alpha = 1/10) + geom_smooth(method = 'lm', formula = y ~ x)

#The model does not fit well, so I will next try and build a polynomial regression model instead.

##########################----------------####################
#Building a polynomial regression model

#Using the below code:
poly_model <- lm(normdata$Price ~ normdata$Beds + normdata$Baths + normdata$Building.Area + normdata$Latitude + normdata$Longitude+I(normdata$Beds^2)+I(normdata$Baths^3)+I(normdata$Building.Area^4)+I(normdata$Latitude^5)+I(normdata$Longitude^6))
summary(poly_model)

#Checking errors
plot(poly_model)

#Now, we can compare the 2 models (linear vs polynomial) together by doing an anova:
anova(linear_model, poly_model, test='F')
#The second model (polynomial) is better!

#To double check, let's plot our curve model on the second poly model
ggplot(data = normdata, aes(x = Beds + Baths + Building.Area + Latitude + Longitude, y = Price)) + geom_point(alpha = 1/10) + geom_smooth(method = 'lm', formula = y ~ x + I(x^2) + I(x^3) + I(x^4) + I(x^5) + I(x^6), color = 'cyan')
#Zooming in:
ggplot(data = normdata, aes(x = Beds + Baths + Building.Area + Latitude + Longitude, y = Price)) + ylim(0, 5) + xlim(-10, 10) + geom_point(alpha = 1/10) +
  labs(title="Polynomial model fitting to dataset") + geom_smooth(method = 'lm', formula = y ~ x + I(x^2) + I(x^3) + I(x^4) + I(x^5) + I(x^6), color = 'cyan')

#################--------------------------##########################
# As the poly model appears to be better, let's try and improve it

#I want to improve my poly model now by using backward elimination to remove the predictor with the largest p-value over 0.05. In this case, I will remove "Beds" & "Longitude" first, then fit the model again.
fit <-  lm(normdata$Price ~ normdata$Baths + normdata$Building.Area + normdata$Latitude  + I(normdata$Baths^2)+I(normdata$Building.Area^3) + I(normdata$Latitude^4))
summary(fit)
plot(fit)

#To double check, let's plot our curve model:
ggplot(data = normdata, aes(x = Baths + Building.Area + Latitude, y = Price)) + geom_point(alpha = 1/10) + geom_smooth(method = 'lm', formula = y ~ x + I(x^2) + I(x^3) + + I(x^4), color = 'cyan')
#Zooming in
ggplot(data = normdata, aes(x = Baths + Building.Area + Latitude, y = Price)) + ylim(0, 5) + xlim(0, 10) + geom_point(alpha = 1/10) + geom_smooth(method = 'lm', formula = y ~ x + I(x^2) + I(x^3) + + I(x^4), color = 'cyan')

#From the last model "fit", I am again using backward elimination to remove the predictor with the largest p-value over 0.05. In this case, I will remove "Latitude", then fit the model again.
fit_ok <- lm(normdata$Price ~ normdata$Baths + normdata$Building.Area  + I(normdata$Baths^2)+I(normdata$Building.Area^3))
summary(fit_ok)
plot(fit_ok)

# Plotting our new curve and model
ggplot(data = normdata, aes(x = Baths + Building.Area, y = Price)) + geom_point(alpha = 1/10) + geom_smooth(method = 'lm', formula = y ~ x + I(x^2) + I(x^3), color = 'cyan')
#Zooming in:
ggplot(data = normdata, aes(x = Baths + Building.Area, y = Price)) + ylim(0, 5) + xlim(-5, 20) + geom_point(alpha = 1/10) + geom_smooth(method = 'lm', formula = y ~ x + I(x^2) + I(x^3), color = 'cyan')

#Now, we can compare these 2 new poly models together by doing an anova:
anova(fit, fit_ok, test='F')
#None is too significant

#Also comparing the initial poly together with these 2 new poly models
anova(poly_model, fit, fit_ok, test='F')
#Fit is slightly better

####################-----------------#######################
#Repeating the normalization and polynomial regression model BUT with the dataset without OUTLIERS: new_df

#Proceeding to standardizing the data with library caret
preprocdata_noout <- preProcess(new_df[,c(2,7:11)], method=c("center", "scale"))
normdata_noout <-predict(preprocdata_noout, new_df[,c(2,7:11)])
summary(normdata_noout)
View(normdata_noout)

#Creating the linear regression model with all variables and with the dataset without outliers
linear_model_nootl <- lm(normdata_noout$Price ~ normdata_noout$Beds + normdata_noout$Baths + normdata_noout$Building.Area + normdata_noout$Latitude + normdata_noout$Longitude)
summary(linear_model_nootl)
#It looks better

#Creating the polynomial regression model with all variables and with the dataset without outliers
poly_model_noout <- lm(normdata_noout$Price ~ normdata_noout$Beds + normdata_noout$Baths + normdata_noout$Building.Area + normdata_noout$Latitude + normdata_noout$Longitude+I(normdata_noout$Beds^2)+I(normdata_noout$Baths^3)+I(normdata_noout$Building.Area^4)+I(normdata_noout$Latitude^5)+I(normdata_noout$Longitude^6))
summary(poly_model_noout)
plot(poly_model_noout)
#It looks even better!

#I want to improve both my linear and my poly model without outliers:

#Using backward elimination to remove the predictor with the largest p-value over 0.05. In this case, I will remove "Building.Area" first, then fit the model again.
linear_fit_noot <- lm(normdata_noout$Price ~ normdata_noout$Beds + normdata_noout$Baths + normdata_noout$Latitude + normdata_noout$Longitude)
summary(linear_fit_noot)
# All significant results!

#Using backward elimination to remove the predictor with the largest p-value over 0.05. In this case, I will remove "Beds" & "Building.Area" first, then fit the model again.
poly_fit_noout <-  lm(normdata_noout$Price ~ normdata_noout$Baths + normdata_noout$Latitude + normdata_noout$Longitude + I(normdata_noout$Baths^2)+I(normdata_noout$Latitude^3)+I(normdata_noout$Longitude^4))
summary(poly_fit_noout)
plot(poly_fit_noout)
#It looks pretty good all variables are significant!

#Now, we can compare all these models together by doing an Anova regression Technique Anova 'F' Test 

#First, comparing both lineal models:
anova(linear_model_nootl, linear_fit_noot, test='F')
#No significant results

#Secondly comparing full linear model with full poly model
anova(linear_model_nootl, poly_model_noout, test='F')
#poly wins with < 2.2e-16 

#Then comparing the two fitted linear and poly models
anova(linear_fit_noot,poly_fit_noout, test='F')
#poly still performing better with < 2.2e-16

#Comparing the 2 poly models together to see which one is the best:
anova(poly_model_noout, poly_fit_noout, test='F')
#No particular significance between the 2 models!

#To double check, let's plot our curve model on the best performing poly_fit_noout 
ggplot(data = normdata_noout, aes(x = Baths + Latitude + Longitude, y = Price)) + geom_point() + geom_smooth(method = 'lm', formula = y ~ x + I(x^2) + I(x^3) +I(x^4), color = 'cyan')

#To double check, let's plot our curve model on the other best performing poly_model_noout
ggplot(data = normdata_noout, aes(x = Baths + Latitude + Longitude, y = Price)) + geom_point() + geom_smooth(method = 'lm', formula = y ~ x + I(x^2) + I(x^3) +I(x^4) +I(x^5) +I(x^6), color = 'cyan')

#################------------------------###########################
#LET'S MAKE SOME PREDICTIONS NOW, FROM THE FULL DATASET

#Let's predict the house price from the models:

#Predict the house price from the best performing Polynomial model with outliers, "fit" model
new <- subset(normdata)
prediction <- predict(fit, newdata = new)
head(prediction)

#Calculating the R-squared for the prediction model on the data set. In general, R-squared is the metric for evaluating the goodness of fit of my model. Higher is better with 1 being the best.
SSE <- sum((new$Price- prediction) ^ 2)
SST <- sum((new$Price - mean(new$Price)) ^ 2)
x = 1 - SSE/SST
x
#It is low

#Predicts the values with confidence interval 
confint(fit, level=0.95)

#Plotting the fitted model versus the residuals. No clear pattern should show in the residual plot if the model is a good fit
plot(fitted(fit),residuals(fit))

#Plotting predicted vs. actual values
plot(x=predict(fit), y=normdata$Price,
     xlab='Predicted Values',
     ylab='Actual Values',
     main='Predicted vs. Actual Values')

#Using model to create prediction intervals
prediction <- predict(fit, interval = "predict")

#Creating dataset that contains original data along with prediction intervals
all_data <- cbind(normdata, prediction)

#Finally creating the comprehensive plot
ggplot(all_data, aes(x = Baths + Building.Area + Latitude, y = Price)) + #define x and y axis variables
  geom_point() +
  labs(title="Fit model in full dataset, within prediction intervals") + 
  stat_smooth(method = lm, formula = y~poly(x,3)) + 
  geom_line(aes(y = lwr), col = "coral2", linetype = "dashed") + #lwr pred interval
  geom_line(aes(y = upr), col = "coral2", linetype = "dashed") #upr pred interval

#### Predict the house price from the second best performing Polynomial model with outliers, "poly_model" 
new <- subset(normdata)
prediction <- predict(poly_model, newdata = new)
head(prediction)

#Calculating of R-squared for the prediction model on the data set. In general, R-squared is the metric for evaluating the goodness of fit of my model. Higher is better with 1 being the best.
SSE <- sum((new$Price- prediction) ^ 2)
SST <- sum((new$Price - mean(new$Price)) ^ 2)
x = 1 - SSE/SST
x
#It is slightly better, but still low

#Predicts the values with confidence interval 
confint(poly_model, level=0.95)

#Plotting predicted vs. actual values
plot(x=predict(poly_model), y=normdata$Price,
     xlab='Predicted Values',
     ylab='Actual Values',
     main='Predicted vs. Actual Values')

#Using model to create prediction intervals
prediction <- predict(poly_model, interval = "predict")

#Creating dataset that contains original data along with prediction intervals
all_data <- cbind(normdata, prediction)

#Finally creating the comprehensive graph
ggplot(all_data, aes(x = Beds + Baths + Building.Area + Latitude + Longitude, y = Price)) + #define x and y axis variables
  geom_point() +
  labs(title="poly_model in full dataset, within prediction intervals") +
  stat_smooth(method = lm, formula = y~poly(x,6)) + 
  geom_line(aes(y = lwr), col = "coral2", linetype = "dashed") + #lwr pred interval
  geom_line(aes(y = upr), col = "coral2", linetype = "dashed") #upr pred interval


#NEXT CHECK: Let's predict the house price from the following models:

#Predicting the house price from the polynomial with outliers, "fit_ok" model
new <- subset(normdata)
prediction2<- predict(fit_ok,  newdata = new)
head(prediction2)

SSE <- sum((new$Price- prediction2) ^ 2)
SST <- sum((new$Price - mean(new$Price)) ^ 2)
x = 1 - SSE/SST
x
#It is even lower

#NEXT CHECK: Predicting the house price from the linear model with outliers
new <- subset(normdata)
prediction4 <- predict(linear_model,  newdata = new)
head(prediction4)

SSE <- sum((new$Price- prediction4) ^ 2)
SST <- sum((new$Price - mean(new$Price)) ^ 2)
x = 1 - SSE/SST
x
#Still low

##################NEXT CHECK -> DATASET WITHOUT OUTLIERS!

#Predicting the house price from the first polymodel without outliers, best performing poly_fit_noout
new_noot <- subset(normdata_noout)
prediction5 <- predict(poly_fit_noout,  newdata = new_noot)
head(prediction5)

#Checking the SSE
SSE <- sum((new_noot$Price- prediction5) ^ 2)
SST <- sum((new_noot$Price - mean(new_noot$Price)) ^ 2)
x = 1 - SSE/SST
x
#Better! Let's do additional checks and plotting:

#Predicts the values with confidence interval 
confint(poly_fit_noout, level=0.95)

#Plotting the fitted model versus the residuals. No clear pattern should show in the residual plot if the model is a good fit
plot(fitted(poly_fit_noout),residuals(poly_fit_noout))

#Plotting predicted vs. actual values
plot(x=predict(poly_fit_noout), y=new_noot$Price,
     xlab='Predicted Values',
     ylab='Actual Values',
     main='Predicted vs. Actual Values')

#Using model to create prediction intervals
prediction5 <- predict(poly_fit_noout, interval = "predict")

#Creating dataset that contains original data along with prediction intervals
all_data <- cbind(normdata_noout, prediction5)

#Finally creating the cumulative plot
ggplot(all_data, aes(x = Baths + Latitude + Longitude, y = Price)) + #define x and y axis variables
  geom_point() + #add scatterplot points
  labs(title="poly_fit_noout model in dataset without outliers, within prediction intervals") +
  stat_smooth(method = lm, formula = y~poly(x,4)) + 
  geom_line(aes(y = lwr), col = "coral2", linetype = "dashed") + #lwr pred interval
  geom_line(aes(y = upr), col = "coral2", linetype = "dashed") #upr pred interval


#NEXT CHECK: Predicting the house price from the polymodel without outliers, second best performing poly_model_noout model

new_noot <- subset(normdata_noout)
prediction6 <- predict(poly_model_noout,  newdata = new_noot)
head(prediction6)

#Checking the SSE
SSE <- sum((new_noot$Price- prediction6) ^ 2)
SST <- sum((new_noot$Price - mean(new_noot$Price)) ^ 2)
x = 1 - SSE/SST
x
#Better! Let's do additional checks and plotting:

#Predicts the values with confidence interval 
confint(poly_model_noout, level=0.95)

#Plotting the fitted model versus the residuals. No clear pattern should show in the residual plot if the model is a good fit
plot(fitted(poly_model_noout),residuals(poly_model_noout))

#Plotting predicted vs. actual values
plot(x=predict(poly_model_noout), y=new_noot$Price,
     xlab='Predicted Values',
     ylab='Actual Values',
     main='Predicted vs. Actual Values')

#Using model to create prediction intervals
prediction6 <- predict(poly_model_noout, interval = "predict")

#Creating dataset that contains original data along with prediction intervals
all_data <- cbind(normdata_noout, prediction6)

#Finally creating the plot
ggplot(all_data, aes(x = Beds + Baths + Building.Area + Latitude + Longitude, y = Price)) + #define x and y axis variables
  geom_point() + #add scatterplot points
  labs(title="poly_model_noout model in dataset without outliers, within prediction intervals") +
  stat_smooth(method = lm, formula = y~poly(x,6)) + 
  geom_line(aes(y = lwr), col = "coral2", linetype = "dashed") + #lwr pred interval
  geom_line(aes(y = upr), col = "coral2", linetype = "dashed") #upr pred interval

#NEXT CHECK, predicting the house price for the linear models:
new_noot <- subset(normdata_noout)
prediction7 <- predict(linear_model_nootl,  newdata = new_noot)
head(prediction7)

SSE <- sum((new_noot$Price- prediction7) ^ 2)
SST <- sum((new_noot$Price - mean(new_noot$Price)) ^ 2)
x = 1 - SSE/SST
x

#NEW CHECK: predicting house proce for linear model linear_fit_noot
new_noot <- subset(normdata_noout)
prediction8 <- predict(linear_fit_noot,  newdata = new_noot)
head(prediction8)

SSE <- sum((new_noot$Price- prediction8) ^ 2)
SST <- sum((new_noot$Price - mean(new_noot$Price)) ^ 2)
x = 1 - SSE/SST
x

###############-----------------#####################
#Now training the 2 best performing models (poly_fit_noout & poly_model_noout), FIRST WITH THE DATASET WITHOUT OUTLIERS
#Starting with model poly_fit_noout
#We will partition into 70% training and 30% validation
set.seed(1)
trainrows <- sample(rownames(normdata_noout), dim(normdata_noout)[1]*0.7)
traindata <- normdata_noout[trainrows, ]
str(traindata)

# we will set the difference of the training into the validation set i.e. 30%
validrows <- setdiff(rownames(normdata_noout), trainrows)
validdata <- normdata_noout[validrows, ]
str(validdata)

# Calculating the regression model on best performing model poly_fit_noout, using my training data
poly_fit_noout_reg <- lm(normdata_noout$Price ~ normdata_noout$Baths + normdata_noout$Latitude + normdata_noout$Longitude + I(normdata_noout$Baths^2)+ I(normdata_noout$Latitude^3)+I(normdata_noout$Longitude^4), data = normdata_noout, subset=trainrows) 
poly_fit_noout_reg

#We will be now predicting the value of the house prices on the training set. 
pred <- predict(poly_fit_noout_reg, traindata = traindata)

#Creating a data frame to show Price predictions and residuals
vres <- data.frame(traindata$Price, pred, residuals = traindata$Price - pred)
head(vres)

#Calculate the  R-squared for the prediction model on the traindata set. 
SSE <- sum((traindata$Price - pred) ^ 2)
SST <- sum((traindata$Price - mean(traindata$Price)) ^ 2)
x = 1 - SSE/SST
x

#Calculating the regression model, nwo using my validation data
poly_fit_noout_reg1 <- lm(normdata_noout$Price ~ normdata_noout$Baths + normdata_noout$Latitude + normdata_noout$Longitude + I(normdata_noout$Baths^2)+ I(normdata_noout$Latitude^3)+I(normdata_noout$Longitude^4), data = normdata_noout, subset=validrows) 
poly_fit_noout_reg1

#Predicting values of house prices on validation data + creating a df to show Price predictions and residuals
pred1 <- predict(poly_fit_noout_reg1, validdata = validdata)
vres1 <- data.frame(validdata$Price, pred1, residuals = validdata$Price - pred1)
head(vres1)

#At last, calculate the validation data of R-squared for the prediction model on the validdata set. 
SSE <- sum((validdata$Price - pred1) ^ 2)
SST <- sum((validdata$Price - mean(validdata$Price)) ^ 2)
x = 1 - SSE/SST
x

#### Now training the with model poly_model_noout model

#We will partition into 70% training and 30% validation
set.seed(1)
trainrows <- sample(rownames(normdata_noout), dim(normdata_noout)[1]*0.7)
traindata <- normdata_noout[trainrows, ]
str(traindata)

# we will set the difference of the training into the validation set i.e. 30%
validrows <- setdiff(rownames(normdata_noout), trainrows)
validdata <- normdata_noout[validrows, ]
str(validdata)

# Calculating the regression model on second best performing model poly_model_noout, using my training data
poly_model_noout_reg <- lm(normdata_noout$Price ~ normdata_noout$Beds + normdata_noout$Baths + normdata_noout$Building.Area + normdata_noout$Latitude + normdata_noout$Longitude+I(normdata_noout$Beds^2)+I(normdata_noout$Baths^3)+I(normdata_noout$Building.Area^4)+I(normdata_noout$Latitude^5)+I(normdata_noout$Longitude^6), data = normdata_noout, subset=trainrows) 
poly_model_noout_reg

#We will be now predicting the value of the house prices on the training set. 
pred <- predict(poly_model_noout_reg, traindata = traindata)

#Creating a data frame to show Price predictions and residuals
vres <- data.frame(traindata$Price, pred, residuals = traindata$Price - pred)
head(vres)

#Calculate the  R-squared for the prediction model on the traindata set.
SSE <- sum((traindata$Price - pred) ^ 2)
SST <- sum((traindata$Price - mean(traindata$Price)) ^ 2)
x = 1 - SSE/SST
x

#Calculating the regression model, using my validation data
poly_model_noout_reg1 <- lm(normdata_noout$Price ~ normdata_noout$Beds + normdata_noout$Baths + normdata_noout$Building.Area + normdata_noout$Latitude + normdata_noout$Longitude+I(normdata_noout$Beds^2)+I(normdata_noout$Baths^3)+I(normdata_noout$Building.Area^4)+I(normdata_noout$Latitude^5)+I(normdata_noout$Longitude^6), data = normdata_noout, subset=validrows) 
poly_model_noout_reg1

# Predicting values of house prices on validation data + creating a df to show Price predictions and residuals
pred1 <- predict(poly_model_noout_reg1, validdata = validdata)
vres1 <- data.frame(validdata$Price, pred1, residuals = validdata$Price - pred1)
head(vres1)

#At last, calculate the validation data of R-squared for the prediction model on the test data set. In general, R-squared is the metric for evaluating the goodness of fit of my model. Higher is better with 1 being the best.
SSE <- sum((validdata$Price - pred1) ^ 2)
SST <- sum((validdata$Price - mean(validdata$Price)) ^ 2)
x = 1 - SSE/SST
x
#Slightly improved

##################------------------########
#Now training the 2 best performing models (fit  & poly_model), WITH THE FULL DATASET comprehensive of outliers
#Let's start with model fit

#We will partition into 70% training and 30% validation
set.seed(1)
trainrows <- sample(rownames(normdata), dim(normdata)[1]*0.7)
traindata <- normdata[trainrows, ]
str(traindata)

# we will set the difference of the training into the validation set i.e. 30%
validrows <- setdiff(rownames(normdata), trainrows)
validdata <- normdata[validrows, ]
str(validdata)

# Calculating the regression model, using my training data
fit_reg <- lm(normdata$Price ~ normdata$Baths + normdata$Building.Area + normdata$Latitude + I(normdata$Baths^2)+I(normdata$Building.Area^3) + I(normdata$Latitude^4), data = normdata, subset=trainrows) 
fit_reg

#We will be now using the machine learning technique to predict the value of the house prices on the training set. 
pred1 <- predict(fit_reg, traindata = traindata)

#Creating a data frame to show Price predictions and residuals
vres1 <- data.frame(traindata$Price, pred1, residuals = traindata$Price - pred1)
head(vres1)

#Calculate the  R-squared for the prediction model on the traindata set.
SSE <- sum((traindata$Price - pred1) ^ 2)
SST <- sum((traindata$Price - mean(traindata$Price)) ^ 2)
x = 1 - SSE/SST
x

# Calculating the regression model, using my validation data
fit_reg1 <- lm(normdata$Price ~ normdata$Baths + normdata$Building.Area + normdata$Latitude + I(normdata$Baths^2)+I(normdata$Building.Area^3) + I(normdata$Latitude^4), data = normdata, subset=validrows) 
fit_reg1

# Predicting values of house prices on validation data + creating a df to show Price predictions and residuals
pred2 <- predict(fit_reg1, validdata = validdata)
vres2 <- data.frame(validdata$Price, pred2, residuals = validdata$Price - pred2)
head(vres2)

#At last, calculate the validation data of R-squared for the prediction model on the test data set. In general, R-squared is the metric for evaluating the goodness of fit of my model. Higher is better with 1 being the best.
SSE <- sum((validdata$Price - pred2) ^ 2)
SST <- sum((validdata$Price - mean(validdata$Price)) ^ 2)
x = 1 - SSE/SST
x
#Improved!

### Let's now do it with the other model poly_model 

#We will partition into 70% training and 30% validation
set.seed(1)
trainrows <- sample(rownames(normdata), dim(normdata)[1]*0.7)
traindata <- normdata[trainrows, ]
str(traindata)

# we will set the difference of the training into the validation set i.e. 30%
validrows <- setdiff(rownames(normdata), trainrows)
validdata <- normdata[validrows, ]
str(validdata)

# Calculating the regression model, using my training data
poly_model_reg <- lm(normdata$Price ~ normdata$Beds + normdata$Baths + normdata$Building.Area + normdata$Latitude + normdata$Longitude+I(normdata$Beds^2)+I(normdata$Baths^3)+I(normdata$Building.Area^4)+I(normdata$Latitude^5)+I(normdata$Longitude^6), data = normdata, subset=trainrows) 
poly_model_reg

#We will be now using the machine learning technique to predict the value of the house prices on the training set. 
pred1 <- predict(poly_model_reg, traindata = traindata)

#Creating a data frame to show Price predictions and residuals
vres1 <- data.frame(traindata$Price, pred1, residuals = traindata$Price - pred1)
head(vres1)

#Calculate the  R-squared for the prediction model on the traindata set.
SSE <- sum((traindata$Price - pred1) ^ 2)
SST <- sum((traindata$Price - mean(traindata$Price)) ^ 2)
x = 1 - SSE/SST
x

# Calculating the regression model, using my validation data
poly_model_reg1 <- lm(normdata$Price ~ normdata$Beds + normdata$Baths + normdata$Building.Area + normdata$Latitude + normdata$Longitude+I(normdata$Beds^2)+I(normdata$Baths^3)+I(normdata$Building.Area^4)+I(normdata$Latitude^5)+I(normdata$Longitude^6), data = normdata, subset=validrows) 
poly_model_reg1

# Predicting values of house prices on validation data + creating a df to show Price predictions and residuals
pred2 <- predict(poly_model_reg1, validdata = validdata)
vres2 <- data.frame(validdata$Price, pred2, residuals = validdata$Price - pred2)
head(vres2)

#At last, calculate the validation data of R-squared for the prediction model on the test data set. In general, R-squared is the metric for evaluating the goodness of fit of my model. Higher is better with 1 being the best.
SSE <- sum((validdata$Price - pred2) ^ 2)
SST <- sum((validdata$Price - mean(validdata$Price)) ^ 2)
x = 1 - SSE/SST
x
#Improved!

################--------------------###############################
# Not particularly happy with the results, so trying a LAST ALTERNATIVE REGRESSION MODEL
# SVM regression on the dataset WITHOUT outlier.

# Setting the seed
set.seed(1)
# We will partition the dataset into 70% training and 30% validation
trainrows <- sample(rownames(normdata_noout), dim(normdata_noout)[1]*0.7)
traindata <- normdata_noout[trainrows, ]
str(traindata)

# We will set the difference of the training into the validation set i.e. 30%
validrows <- setdiff(rownames(normdata_noout), trainrows)
validdata <- normdata_noout[validrows, ]
str(validdata)

#Calculating the SVM regression model
model_reg = svm(Price~., data=traindata)
print(model_reg)

#Next, we'll predict the valid data and plot the results to compare visually.
pred_svm = predict(model_reg,validdata)

x=1:length(validdata$Price)
plot(x, validdata$Price, pch=18, col="red")
lines(x, pred_svm, lwd="1", col="blue") 

# Accuracy check 
mse = mse(validdata$Price, pred_svm)
mae = MAE(validdata$Price, pred_svm)
rmse = RMSE(validdata$Price, pred_svm)
r2 = R2(validdata$Price, pred_svm)

cat(" MAE:", mae, "\n", "MSE:", mse, "\n", 
    "RMSE:", rmse, "\n", "R-squared:", r2)
#Better R^2!


