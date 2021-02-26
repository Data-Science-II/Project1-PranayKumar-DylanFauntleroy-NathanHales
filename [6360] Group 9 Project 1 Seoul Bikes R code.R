##Imports and Packages
library(ISLR)
##install.packages("glmnet")
library(glmnet)
##install.packages("dplyr")
library(dplyr)
##install.packages("tidyr")
library(tidyr)
##install.packages("tidyverse")
library(tidyverse)
##install.packages("ggpubr")
library(ggpubr)
##install.packages("rstatix")
library(rstatix)
##install.packages("broom")
library(broom)
library(ISLR)
##install.packages("leaps")
library(leaps)
library(ggplot2)
library(readxl)
##Seoul Bike Dataset
Bike_Data = read.csv("Datasets/SeoulBikeData.csv")
BikesRented = Bike_Data$Rented.Bike.Count
hour = Bike_Data$Hour
temp = Bike_Data$Temperature..C.
humidity = Bike_Data$Humidity...
wind = Bike_Data$Wind.speed..m.s.
visibility = Bike_Data$Visibility..10m.
dew = Bike_Data$Dew.point.temperature..C.
sun = Bike_Data$Solar.Radiation..MJ.m2.
rain = Bike_Data$Rainfall.mm.
snow = Bike_Data$Snowfall..cm.
bike_fit = lm(BikesRented ~ hour+temp+humidity+wind+visibility+dew+sun+rain+snow)
emptyModelFit = lm(BikesRented~1,) 
##Forward Selection Variable Selection
step(emptyModelFit, scope=list(upper= bike_fit), direction="forward")
##hour + temp + humidity + visibility + sun + rain + snow
##BackWard Elimination Variable Selection
step(bike_fit, method = "backward")
##hour + temp + humidity + visibility + sun + rain + snow
##Stepwise Regression Variable Selection
step(emptyModelFit, scope=list(upper= bike_fit), direction="both")
##hour + temp + humidity + visibility + sun + rain + snow
##R-Squared and AIC plots 
other_bike_fit = regsubsets(Bike_Data$Rented.Bike.Count ~ Bike_Data$Hour + Bike_Data$Temperature..C. + Bike_Data$Humidity... + Bike_Data$Wind.speed..m.s. + Bike_Data$Visibility..10m. + Bike_Data$Dew.point.temperature..C. + Bike_Data$Solar.Radiation..MJ.m2. + Bike_Data$Rainfall.mm. + Bike_Data$Snowfall..cm., data = Bike_Data)
plot(other_bike_fit, scale="r2")
plot(other_bike_fit, scale="Cp")
plot(other_bike_fit, scale="adjr2")
plot(other_bike_fit)
##Ridge Regression Variable Selection
x = model.matrix(BikesRented ~ hour + temp + humidity + wind + visibility + dew + sun + rain + snow)
y = Bike_Data$Rented.Bike.Count
grid = 10^seq(10, -2, length = 100)
ridge_mod = glmnet(x, y, alpha = 0, lambda = grid)
dim(coef(ridge_mod))
plot(ridge_mod)
set.seed(1)
train=sample(1:nrow(x), nrow(x)/2)
test=(-train)
y.test=y[test]
cv.out=cv.glmnet(x[train ,],y[train],alpha=0)
plot(cv.out)
bestlam=cv.out$lambda.min
bestlam
ridge_pred=predict(ridge_mod,s=bestlam ,newx=x[test,])
MSE = mean((ridge_pred-y.test)^2)
sqrt(MSE)
out=glmnet(x,y,alpha=0)
predict(out,type="coefficients",s=bestlam)
##hour + temp + humidity + wind + visibility + dew + sun + rain + snow
##Lasso Regression Variable Selection
lasso_mod=glmnet(x[train ,],y[train],alpha=1,lambda=grid)
plot(lasso_mod)
set.seed(1)
cv.out=cv.glmnet(x[train ,],y[train],alpha=1)
plot(cv.out)
bestlam2=cv.out$lambda.min
bestlam2
lasso.pred=predict(lasso_mod,s=bestlam2,newx=x[test,])
MSE_L = mean((lasso.pred-y.test)^2) 
sqrt(MSE_L)
out=glmnet(x,y,alpha=1,lambda=grid)
lasso.coef=predict(out,type="coefficients",s=bestlam2)
lasso.coef
##hour + temp + humidity + wind + visibility + dew + sun + rain + snow

##Multiple Linear Regression using Forward Selection Variables
mlr.model = lm(formula = BikesRented ~ hour + temp + humidity + visibility + sun + rain + snow, data = Bike_Data)
summary(mlr.model)
##Multiple Linear Regression using Backward Elimination Variables
mlr.model = lm(formula = BikesRented ~ hour + temp + humidity + visibility + sun + rain + snow, data = Bike_Data)
summary(mlr.model)
##Multiple Linear Regression using Stepwise Regression Variables
mlr.model = lm(formula = BikesRented ~ hour + temp + humidity + visibility + sun + rain + snow, data = Bike_Data)
summary(mlr.model)
##Multiple Linear Regression using Ridge Regression Variables
mlr.model = lm(formula = BikesRented ~ hour + temp + humidity + wind + visibility + dew + sun + rain + snow, data = Bike_Data)
summary(mlr.model)
##Multiple Linear Regression using Lasso Regression Variables
mlr.model = lm(formula = BikesRented ~ hour + temp + humidity + wind + visibility + dew + sun + rain + snow, data = Bike_Data)
summary(mlr.model)
##Quadratic Regression using Forward Selection Variables
quadratic.model = lm(formula = BikesRented ~ poly(hour, 2) + poly(temp, 2) + poly(humidity, 2) + poly(visibility, 2) + poly(sun, 2) + poly(rain, 2) + poly(snow, 2), data = Bike_Data)
summary(quadratic.model)
##Quadratic Regression using Backward Elimination Variables
quadratic.model = lm(formula = BikesRented ~ poly(hour, 2) + poly(temp, 2) + poly(humidity, 2) + poly(visibility, 2) + poly(sun, 2) + poly(rain, 2) + poly(snow, 2), data = Bike_Data)
summary(quadratic.model)
##Quadratic Regression using Stepwise Regression Variables
quadratic.model = lm(formula = BikesRented ~ poly(hour, 2) + poly(temp, 2) + poly(humidity, 2) + poly(visibility, 2) + poly(sun, 2) + poly(rain, 2) + poly(snow, 2), data = Bike_Data)
summary(quadratic.model)
##Quadratic Regression using Ridge Regression Variables
quadratic.model = lm(formula = BikesRented ~ poly(hour, 2) + poly(temp, 2) + poly(humidity, 2) + poly(wind, 2) + poly(visibility, 2) + poly(dew, 2) + poly(sun, 2) + poly(rain, 2) + poly(snow, 2), data = Bike_Data)
summary(quadratic.model)
##Quadratic Regression using Lasso Regression Variables
quadratic.model = lm(formula = BikesRented ~ poly(hour, 2) + poly(temp, 2) + poly(humidity, 2) + poly(wind, 2) + poly(visibility, 2) + poly(dew, 2) + poly(sun, 2) + poly(rain, 2) + poly(snow, 2), data = Bike_Data)
summary(quadratic.model)
##Cubic Regression using Forward Selection Variables
cubic.model = lm(formula = BikesRented ~ poly(hour, 3) + poly(temp, 3) + poly(humidity, 3) + poly(visibility, 3) + poly(sun, 3) + poly(rain, 3) + poly(snow, 3), data = Bike_Data)
summary(cubic.model)
##Cubic Regression using Backward Elimination Variables
cubic.model = lm(formula = BikesRented ~ poly(hour, 3) + poly(temp, 3) + poly(humidity, 3) + poly(visibility, 3) + poly(sun, 3) + poly(rain, 3) + poly(snow, 3), data = Bike_Data)
summary(cubic.model)
##Cubic Regression using Stepwise Regression Variables
cubic.model = lm(formula = BikesRented ~ poly(hour, 3) + poly(temp, 3) + poly(humidity, 3) + poly(visibility, 3) + poly(sun, 3) + poly(rain, 3) + poly(snow, 3), data = Bike_Data)
summary(cubic.model)
##Cubic Regression using Ridge Regression Variables
cubic.model = lm(formula = BikesRented ~ poly(hour, 3) + poly(temp, 3) + poly(humidity, 3) + poly(wind, 3) + poly(visibility, 3) + poly(dew, 3) + poly(sun, 3) + poly(rain, 3) + poly(snow, 3), data = Bike_Data)
summary(cubic.model)
##Cubic Regression using Lasso Regression Variables
cubic.model = lm(formula = BikesRented ~ poly(hour, 3) + poly(temp, 3) + poly(humidity, 3) + poly(wind, 3) + poly(visibility, 3) + poly(dew, 3) + poly(sun, 3) + poly(rain, 3) + poly(snow, 3), data = Bike_Data)
summary(cubic.model)
##ANCOVA using Forward Selection Variables
ancova.model = aov(formula = BikesRented ~ hour * temp * humidity * visibility * sun * rain * snow, data = Bike_Data)
summary(ancova.model)
##ANCOVA using Backward Elimination Variables
ancova.model = aov(formula = BikesRented ~ hour * temp * humidity * visibility * sun * rain * snow, data = Bike_Data)
summary(ancova.model)
##ANCOVA using Stepwise Regression Variables
ancova.model = aov(formula = BikesRented ~ hour * temp * humidity * visibility * sun * rain * snow, data = Bike_Data)
summary(ancova.model)
##ANCOVA using Ridge Regression Variables
ancova.model = aov(formula = BikesRented ~ hour * temp * humidity * wind * visibility * dew * sun * rain * snow, data = Bike_Data)
summary(ancova.model)
##ANCOVA using Lasso Regression Variables
ancova.model = aov(formula = BikesRented ~ hour * temp * humidity * wind * visibility * dew * sun * rain * snow, data = Bike_Data)
summary(ancova.model)