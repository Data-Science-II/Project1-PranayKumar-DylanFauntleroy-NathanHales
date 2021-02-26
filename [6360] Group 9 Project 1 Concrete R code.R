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
##Concrete Dataset
Concrete_Data = read_excel("Datasets/Concrete_Data.xls")
cement = Concrete_Data$`Cement (component 1)(kg in a m^3 mixture)`
BFS = Concrete_Data$`Blast Furnace Slag (component 2)(kg in a m^3 mixture)`
FlyAsh = Concrete_Data$`Fly Ash (component 3)(kg in a m^3 mixture)`
water = Concrete_Data$`Water  (component 4)(kg in a m^3 mixture)`
superplasticizer = Concrete_Data$`Superplasticizer (component 5)(kg in a m^3 mixture)`
coarseAggregate = Concrete_Data$`Coarse Aggregate  (component 6)(kg in a m^3 mixture)`
fineAggregate = Concrete_Data$`Fine Aggregate (component 7)(kg in a m^3 mixture)`
age = Concrete_Data$`Age (day)`
CCS = Concrete_Data$`Concrete compressive strength(MPa, megapascals)`
concrete_fit = lm(CCS ~ cement + BFS + FlyAsh + water + superplasticizer + coarseAggregate + fineAggregate + age)
emptyModelFit = lm(CCS~1,) 
##Forward Selection Variable Selection
step(emptyModelFit, scope=list(upper= concrete_fit), direction="forward")
##Cement, superplasticizer, age, BFS, water, FlyAsh
##BackWard Elimination Variable Selection
step(concrete_fit, method = "backward")
##Cement, BFS, FlyAsh, water, superplasticizer, coarseAggregate, fineAggregate, Age
##Stepwise Regression Variable Selection
step(emptyModelFit, scope=list(upper= concrete_fit), direction="both")
##cement, superplasticizer, age, BFS, water, FlyAsh
##R-Squared and AIC plots 
other_concrete_fit = regsubsets(Concrete_Data$`Concrete compressive strength(MPa, megapascals)` ~ Concrete_Data$`Cement (component 1)(kg in a m^3 mixture)` + Concrete_Data$`Blast Furnace Slag (component 2)(kg in a m^3 mixture)` + Concrete_Data$`Fly Ash (component 3)(kg in a m^3 mixture)` + Concrete_Data$`Water  (component 4)(kg in a m^3 mixture)` + Concrete_Data$`Superplasticizer (component 5)(kg in a m^3 mixture)` + Concrete_Data$`Coarse Aggregate  (component 6)(kg in a m^3 mixture)` + Concrete_Data$`Fine Aggregate (component 7)(kg in a m^3 mixture)` + Concrete_Data$`Age (day)`, data = Concrete_Data)
plot(other_concrete_fit, scale="r2")
plot(other_concrete_fit, scale="Cp")
plot(other_concrete_fit, scale="adjr2") 
plot(other_concrete_fit)
##Ridge Regression Variable Selection
x = model.matrix(CCS ~ cement + BFS + FlyAsh + water + superplasticizer + coarseAggregate + fineAggregate + age)
y = Concrete_Data$`Concrete compressive strength(MPa, megapascals)`
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
##Cement, BFS, FlyAsh, water, superplasticizer, coarseAggregate, fineAggregate, Age
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
##Cement, BFS, FlyAsh, Water, Superplasticizer, coarseAggregate, age

##Multiple Linear Regression using Forward Selection Variables
mlr.model = lm(formula = CCS ~ cement + BFS + FlyAsh + water + superplasticizer + age, data = Concrete_Data)
summary(mlr.model)
##Multiple Linear Regression using Backward Elimination Variables
mlr.model = lm(formula = CCS ~ cement + BFS + FlyAsh + water + superplasticizer + age + coarseAggregate + fineAggregate, data = Concrete_Data)
summary(mlr.model)
##Multiple Linear Regression using Stepwise Regression Variables
mlr.model = lm(formula = CCS ~ cement + BFS + FlyAsh + water + superplasticizer + age, data = Concrete_Data)
summary(mlr.model)
##Multiple Linear Regression using Ridge Regression Variables
mlr.model = lm(formula = CCS ~ cement + BFS + FlyAsh + water + superplasticizer + age + coarseAggregate + fineAggregate, data = Concrete_Data)
summary(mlr.model)
##Multiple Linear Regression using Lasso Regression Variables
mlr.model = lm(formula = CCS ~ cement + BFS + FlyAsh + water + superplasticizer + age + coarseAggregate, data = Concrete_Data)
summary(mlr.model)
##Quadratic Regression using Forward Selection Variables
quadratic.model = lm(formula = CCS ~ poly(cement, 2) + poly(BFS, 2) + poly(FlyAsh, 2) + poly(water, 2) + poly(superplasticizer, 2) + poly(age, 2), data = Concrete_Data)
summary(quadratic.model)
##Quadratic Regression using Backward Elimination Variables
quadratic.model = lm(formula = CCS ~ poly(cement, 2) + poly(BFS, 2) + poly(FlyAsh, 2) + poly(water, 2) + poly(superplasticizer, 2) + poly(coarseAggregate, 2) + poly(fineAggregate, 2) + poly(age, 2), data = Concrete_Data)
summary(quadratic.model)
##Quadratic Regression using Stepwise Regression Variables
quadratic.model = lm(formula = CCS ~ poly(cement, 2) + poly(BFS, 2) + poly(FlyAsh, 2) + poly(water, 2) + poly(superplasticizer, 2) + poly(age, 2), data = Concrete_Data)
summary(quadratic.model)
##Quadratic Regression using Ridge Regression Variables
quadratic.model = lm(formula = CCS ~ poly(cement, 2) + poly(BFS, 2) + poly(FlyAsh, 2) + poly(water, 2) + poly(superplasticizer, 2) + poly(coarseAggregate, 2) + poly(fineAggregate, 2) + poly(age, 2), data = Concrete_Data)
summary(quadratic.model)
##Quadratic Regression using Lasso Regression Variables
quadratic.model = lm(formula = CCS ~ poly(cement, 2) + poly(BFS, 2) + poly(FlyAsh, 2) + poly(water, 2) + poly(superplasticizer, 2) + poly(coarseAggregate, 2) + poly(age, 2), data = Concrete_Data)
summary(quadratic.model)
##Cubic Regression using Forward Selection Variables
cubic.model = lm(formula = CCS ~ poly(cement, 3) + poly(BFS, 3) + poly(FlyAsh, 3) + poly(water, 3) + poly(superplasticizer, 3) + poly(age, 3), data = Concrete_Data)
summary(cubic.model)
##Cubic Regression using Backward Elimination Variables
cubic.model = lm(formula = CCS ~ poly(cement, 3) + poly(BFS, 3) + poly(FlyAsh, 3) + poly(water, 3) + poly(superplasticizer, 3) + poly(coarseAggregate, 3) + poly(fineAggregate, 3) + poly(age, 3), data = Concrete_Data)
summary(cubic.model)
##Cubic Regression using Stepwise Regression Variables
cubic.model = lm(formula = CCS ~ poly(cement, 3) + poly(BFS, 3) + poly(FlyAsh, 3) + poly(water, 3) + poly(superplasticizer, 3) + poly(age, 3), data = Concrete_Data)
summary(cubic.model)
##Cubic Regression using Ridge Regression Variables
cubic.model = lm(formula = CCS ~ poly(cement, 3) + poly(BFS, 3) + poly(FlyAsh, 3) + poly(water, 3) + poly(superplasticizer, 3) + poly(coarseAggregate, 3) + poly(fineAggregate, 3) + poly(age, 3), data = Concrete_Data)
summary(cubic.model)
##Cubic Regression using Lasso Regression Variables
cubic.model = lm(formula = CCS ~ poly(cement, 3) + poly(BFS, 3) + poly(FlyAsh, 3) + poly(water, 3) + poly(superplasticizer, 3) + poly(coarseAggregate, 3) + poly(age, 3), data = Concrete_Data)
summary(cubic.model)
##ANCOVA using Forward Selection Variables
ancova.model = aov(formula = CCS ~ cement * BFS * FlyAsh * water * superplasticizer * age, data = Concrete_Data)
summary(ancova.model)
##ANCOVA using Backward Elimination Variables
ancova.model = aov(formula = CCS ~ cement * BFS * FlyAsh * water * superplasticizer * coarseAggregate * fineAggregate * age, data = Concrete_Data)
summary(ancova.model)
##ANCOVA using Stepwise Regression Variables
ancova.model = aov(formula = CCS ~ cement * BFS * FlyAsh * water * superplasticizer * age, data = Concrete_Data)
summary(ancova.model)
##ANCOVA using Ridge Regression Variables
ancova.model = aov(formula = CCS ~ cement * BFS * FlyAsh * water * superplasticizer * coarseAggregate * fineAggregate * age, data = Concrete_Data)
summary(ancova.model)
##ANCOVA using Lasso Regression Variables
ancova.model = aov(formula = CCS ~ cement * BFS * FlyAsh * water * superplasticizer * coarseAggregate * age, data = Concrete_Data)
summary(ancova.model)