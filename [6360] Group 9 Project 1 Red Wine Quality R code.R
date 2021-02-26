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
##Red Wine Quality Dataset
RedWine_Data = read.csv("Datasets/Red_Wine_Quality.csv")
quality = RedWine_Data$quality
fixedacidity = RedWine_Data$fixed.acidity
volatileacidity = RedWine_Data$volatile.acidity
citric = RedWine_Data$citric.acid
sugar = RedWine_Data$residual.sugar
chlorides = RedWine_Data$chlorides
freesulfur = RedWine_Data$free.sulfur.dioxide
totalsulfur = RedWine_Data$total.sulfur.dioxide
density = RedWine_Data$density
ph = RedWine_Data$pH
sulphates = RedWine_Data$sulphates
alcohol = RedWine_Data$alcohol
wine_fit = lm(quality ~ fixedacidity + volatileacidity + citric + sugar + chlorides + freesulfur + totalsulfur + density + ph + sulphates + alcohol)
summary(wine_fit)
emptyModelFit = lm(quality~1,) 
##Forward Selection Variable Selection
step(emptyModelFit, scope=list(upper= wine_fit), direction="forward")
##alcohol + volatileacidity + sulphates + totalsulfur + chlorides + ph + freesulfur
##BackWard Elimination Variable Selection
step(wine_fit, method = "backward")
##alcohol + volatileacidity + sulphates + totalsulfur + chlorides + ph + freesulfur
##Stepwise Regression Variable Selection
step(emptyModelFit, scope=list(upper= wine_fit), direction="both")
##alcohol + volatileacidity + sulphates + totalsulfur + chlorides + ph + freesulfur
##R-Squared and AIC plots 
other_wine_fit = regsubsets(RedWine_Data$quality ~ RedWine_Data$fixed.acidity + RedWine_Data$volatile.acidity + RedWine_Data$citric.acid + RedWine_Data$residual.sugar + RedWine_Data$chlorides + RedWine_Data$free.sulfur.dioxide + RedWine_Data$total.sulfur.dioxide + RedWine_Data$density + RedWine_Data$pH + RedWine_Data$sulphates + RedWine_Data$alcohol, data = RedWine_Data)
plot(other_wine_fit, scale="r2")
plot(other_wine_fit, scale="Cp")
plot(other_wine_fit, scale="adjr2")
plot(other_wine_fit)
##Ridge Regression Variable Selection
x = model.matrix(quality ~ fixedacidity + volatileacidity + citric + sugar + chlorides + freesulfur + totalsulfur + density + ph + sulphates + alcohol)
y = RedWine_Data$quality
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
##fixedacidity + volatileacidity + citric + sugar + chlorides + freesulfur + totalsulfur + density + ph + sulphates + alcohol
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
##fixedacidity + volatileacidity + chlorides + totalsulfur + ph + sulphates + alcohol

##Multiple Linear Regression using Forward Selection Variables
mlr.model = lm(formula = quality ~ alcohol + volatileacidity + sulphates + totalsulfur + chlorides + ph + freesulfur, data = RedWine_Data)
summary(mlr.model)
##Multiple Linear Regression using Backward Elimination Variables
mlr.model = lm(formula = quality ~ alcohol + volatileacidity + sulphates + totalsulfur + chlorides + ph + freesulfur, data = RedWine_Data)
summary(mlr.model)
##Multiple Linear Regression using Stepwise Regression Variables
mlr.model = lm(formula = quality ~ alcohol + volatileacidity + sulphates + totalsulfur + chlorides + ph + freesulfur, data = RedWine_Data)
summary(mlr.model)
##Multiple Linear Regression using Ridge Regression Variables
mlr.model = lm(formula = quality ~ fixedacidity + volatileacidity + citric + sugar + chlorides + freesulfur + totalsulfur + density + ph + sulphates + alcohol, data = RedWine_Data)
summary(mlr.model)
##Multiple Linear Regression using Lasso Regression Variables
mlr.model = lm(formula = quality ~ fixedacidity + volatileacidity + chlorides + totalsulfur + ph + sulphates + alcohol, data = RedWine_Data)
summary(mlr.model)
##Quadratic Regression using Forward Selection Variables
quadratic.model = lm(formula = quality ~ poly(alcohol, 2) + poly(volatileacidity, 2) + poly(sulphates, 2) + poly(totalsulfur, 2) + poly(chlorides, 2) + poly(ph, 2) + poly(freesulfur, 2), data = RedWine_Data)
summary(quadratic.model)
##Quadratic Regression using Backward Elimination Variables
quadratic.model = lm(formula = quality ~ poly(alcohol, 2) + poly(volatileacidity, 2) + poly(sulphates, 2) + poly(totalsulfur, 2) + poly(chlorides, 2) + poly(ph, 2) + poly(freesulfur, 2), data = RedWine_Data)
summary(quadratic.model)
##Quadratic Regression using Stepwise Regression Variables
quadratic.model = lm(formula = quality ~ poly(alcohol, 2) + poly(volatileacidity, 2) + poly(sulphates, 2) + poly(totalsulfur, 2) + poly(chlorides, 2) + poly(ph, 2) + poly(freesulfur, 2), data = RedWine_Data)
summary(quadratic.model)
##Quadratic Regression using Ridge Regression Variables
quadratic.model = lm(formula = quality ~ poly(alcohol, 2) + poly(volatileacidity, 2) + poly(sulphates, 2) + poly(totalsulfur, 2) + poly(chlorides, 2) + poly(ph, 2) + poly(freesulfur, 2) + poly(density, 2) + poly(citric, 2) + poly(sugar, 2) + poly(fixedacidity, 2), data = RedWine_Data)
summary(quadratic.model)
##Quadratic Regression using Lasso Regression Variables
quadratic.model = lm(formula = quality ~ poly(alcohol, 2) + poly(volatileacidity, 2) + poly(sulphates, 2) + poly(totalsulfur, 2) + poly(chlorides, 2) + poly(ph, 2) + poly(fixedacidity, 2), data = RedWine_Data)
summary(quadratic.model)
##Cubic Regression using Forward Selection Variables
cubic.model = lm(formula = quality ~ poly(alcohol, 3) + poly(volatileacidity, 3) + poly(sulphates, 3) + poly(totalsulfur, 3) + poly(chlorides, 3) + poly(ph, 3) + poly(freesulfur, 3), data = RedWine_Data)
summary(cubic.model)
##Cubic Regression using Backward Elimination Variables
cubic.model = lm(formula = quality ~ poly(alcohol, 3) + poly(volatileacidity, 3) + poly(sulphates, 3) + poly(totalsulfur, 3) + poly(chlorides, 3) + poly(ph, 3) + poly(freesulfur, 3), data = RedWine_Data)
summary(cubic.model)
##Cubic Regression using Stepwise Regression Variables
cubic.model = lm(formula = quality ~ poly(alcohol, 3) + poly(volatileacidity, 3) + poly(sulphates, 3) + poly(totalsulfur, 3) + poly(chlorides, 3) + poly(ph, 3) + poly(freesulfur, 3), data = RedWine_Data)
summary(cubic.model)
##Cubic Regression using Ridge Regression Variables
cubic.model = lm(formula = quality ~ poly(alcohol, 3) + poly(volatileacidity, 3) + poly(sulphates, 3) + poly(totalsulfur, 3) + poly(chlorides, 3) + poly(ph, 3) + poly(freesulfur, 3) + poly(density, 3) + poly(citric, 3) + poly(sugar, 3) + poly(fixedacidity, 3), data = RedWine_Data)
summary(cubic.model)
##Cubic Regression using Lasso Regression Variables
cubic.model = lm(formula = quality ~ poly(alcohol, 3) + poly(volatileacidity, 3) + poly(sulphates, 3) + poly(totalsulfur, 3) + poly(chlorides, 3) + poly(ph, 3) + poly(fixedacidity, 3), data = RedWine_Data)
summary(cubic.model)