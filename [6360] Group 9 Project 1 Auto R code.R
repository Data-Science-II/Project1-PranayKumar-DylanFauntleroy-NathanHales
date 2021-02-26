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
##Auto Dataset
fit1 = lm(mpg ~ cylinders + displacement + horsepower + weight + acceleration + year, data = Auto)
emptyModelFit = lm(mpg~1, data=Auto) 
##Backward Selection
step(fit1, method = "backward")
##Weight and Year
##Forward
step(emptyModelFit, scope=list(upper= fit1), direction="forward")
##Weight and Year
##Stepwise
step(emptyModelFit, scope=list(upper= fit1), direction="both")
##Weight and Year
##R^2 and AIC Plots
otherfit = regsubsets(mpg ~ cylinders + displacement + horsepower + weight + acceleration + year, data = Auto)
plot(otherfit, scale="r2")
plot(otherfit, scale="Cp")
plot(otherfit, scale="adjr2")
plot(otherfit)
##Ridge Regression Variable Selection
x = model.matrix(mpg~cylinders + displacement + horsepower + weight + acceleration + year,Auto)
y = Auto$mpg
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
##Uses all 6 variables
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
##Uses Cylinders, weight, acceleration, and year

##Multiple Linear Regression using Forward Selection Variables
mlr.model = lm(formula = mpg ~ weight + year, data = Auto)
summary(mlr.model)
##Multiple Linear Regression using Backward Elimination Variables
mlr.model = lm(formula = mpg ~ weight + year, data = Auto)
summary(mlr.model)
##Multiple Linear Regression using Stepwise Regression Variables
mlr.model = lm(formula = mpg ~ weight + year, data = Auto)
summary(mlr.model)
##Multiple Linear Regression using Ridge Regression Variables
mlr.model = lm(formula = mpg ~ weight + year+displacement+cylinders+horsepower+acceleration, data = Auto)
summary(mlr.model)
##Multiple Linear Regression using Lasso Regression Variables
mlr.model = lm(formula = mpg ~ weight + year +cylinders+acceleration, data = Auto)
summary(mlr.model)
##Quadratic Regression using Forward Selection Variables 
quadratic.model = lm(formula = mpg ~ poly(weight, 2) + poly(year, 2), data = Auto)
summary(quadratic.model)
##Quadratic Regression using Backward Elimination Variables
quadratic.model = lm(formula = mpg ~ poly(weight, 2) + poly(year, 2), data = Auto)
summary(quadratic.model)
##Quadratic Regression using Stepwise Regression Variables
quadratic.model = lm(formula = mpg ~ poly(weight, 2) + poly(year, 2), data = Auto)
summary(quadratic.model)
##Quadratic Regression using Ridge Regression Variables
quadratic.model = lm(formula = mpg ~ poly(weight, 2) + poly(year, 2) + poly(cylinders, 2) + poly(displacement, 2) + poly(horsepower, 2) + poly(acceleration, 2), data = Auto)
summary(quadratic.model)
##Quadratic Regression using Lasso Regression Variables
quadratic.model = lm(formula = mpg ~ poly(weight, 2) + poly(year, 2) + poly(cylinders, 2)  + poly(acceleration, 2), data = Auto)
summary(quadratic.model)
##Cubic Regression using Forward Selection Variables
cubic.model = lm(formula = mpg ~ poly(weight, 3) + poly(year, 3), data = Auto)
summary(cubic.model)
##Cubic Regression using Backward Elimination Variables
cubic.model = lm(formula = mpg ~ poly(weight, 3) + poly(year, 3), data = Auto)
summary(cubic.model)
##Cubic Regression using Stepwise Regression Variables
cubic.model = lm(formula = mpg ~ poly(weight, 3) + poly(year, 3), data = Auto)
summary(cubic.model)
##Cubic Regression using Ridge Regression Variables
cubic.model = lm(formula = mpg ~ poly(weight, 3) + poly(year, 3) + poly(cylinders, 3) + poly(displacement, 3) + poly(horsepower, 3) + poly(acceleration, 3), data = Auto)
summary(cubic.model)
##Cubic Regression using Lasso Regression Variables
cubic.model = lm(formula = mpg ~ poly(weight, 3) + poly(year, 3) + poly(cylinders, 3)  + poly(acceleration, 3), data = Auto)
summary(cubic.model)
##ANCOVA using Forward Selection Variables
ancova.model = aov(formula = mpg ~ weight * year, data = Auto)
summary(ancova.model)
##ANCOVA using Backward Elimination Variables
ancova.model = aov(formula = mpg ~ weight * year, data = Auto)
summary(ancova.model)
##ANCOVA using Stepwise Regression Variables
ancova.model = aov(formula = mpg ~ weight * year, data = Auto)
summary(ancova.model)
##ANCOVA using Ridge Regression Variables
ancova.model = aov(formula = mpg ~ weight * year*displacement*cylinders*horsepower*acceleration, data = Auto)
summary(ancova.model)
##ANCOVA using Lasso Regression Variables
ancova.model = aov(formula = mpg ~ weight * year*cylinders*acceleration, data = Auto)
summary(ancova.model)