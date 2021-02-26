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
##Stability Dataset
Stability_Data = read.csv("C:/Users/hales/OneDrive/Documents/Data Science II/Data_for_UCI_named.csv")
stab = Stability_Data$stab
tau1 = Stability_Data$tau1
tau2 = Stability_Data$tau2
tau3 = Stability_Data$tau3
tau4 = Stability_Data$tau4
p1 = Stability_Data$p1
p2 = Stability_Data$p2
p3 = Stability_Data$p3
g1 = Stability_Data$g1
g2 = Stability_Data$g2
g3 = Stability_Data$g3
g4 = Stability_Data$g4
stability_fit = lm(stab ~ tau1 + tau2 + tau3 + tau4 + p1 + p2 + p3 + g1 + g2 + g3 + g4)
summary(stability_fit)
emptyModelFit = lm(stab~1,) 
##Forward Selection Variable Selection
step(emptyModelFit, scope=list(upper= stability_fit), direction="forward")
##g1 + g2 + g3 + g4 + tau1 + tau2 + tau3 + tau4  + p3
##BackWard Elimination Variable Selection
step(stability_fit, method = "backward")
##g1 + g2 + g3 + g4 + tau1 + tau2 + tau3 + tau4  + p3
##Stepwise Regression Variable Selection
step(emptyModelFit, scope=list(upper= stability_fit), direction="both")
##g1 + g2 + g3 + g4 + tau1 + tau2 + tau3 + tau4  + p3
##R-Squared and AIC plots 
other_stability_fit = regsubsets(Stability_Data$stab ~ Stability_Data$tau1 + Stability_Data$tau2 + Stability_Data$tau3 + Stability_Data$tau4 + Stability_Data$p1 + Stability_Data$p2 + Stability_Data$p3 + Stability_Data$g1 + Stability_Data$g2 + Stability_Data$g3 + Stability_Data$g4, data = Stability_Data)
plot(other_stability_fit, scale="r2")
plot(other_stability_fit, scale="Cp")
plot(other_stability_fit, scale="adjr2") 
plot(other_stability_fit)
##Ridge Regression Variable Selection
x = model.matrix(stab ~ g1 + g2 + g3 + g4 + tau1 + tau2 + tau3 + tau4  + p1 + p2 + p3)
y = Stability_Data$stab
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
##g1 + g2 + g3 + g4 + tau1 + tau2 + tau3 + tau4  + p1 + p2 + p3
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
##g1 + g2 + g3 + g4 + tau1 + tau2 + tau3 + tau4

##Multiple Linear Regression using Forward Selection Variables
mlr.model = lm(formula = stab ~ g1 + g2 + g3 + g4 + tau1 + tau2 + tau3 + tau4  + p3, data = Stability_Data)
summary(mlr.model)
##Multiple Linear Regression using Backward Elimination Variables
mlr.model = lm(formula = stab ~ g1 + g2 + g3 + g4 + tau1 + tau2 + tau3 + tau4  + p3, data = Stability_Data)
summary(mlr.model)
##Multiple Linear Regression using Stepwise Regression Variables
mlr.model = lm(formula = stab ~ g1 + g2 + g3 + g4 + tau1 + tau2 + tau3 + tau4  + p3, data = Stability_Data)
summary(mlr.model)
##Multiple Linear Regression using Ridge Regression Variables
mlr.model = lm(formula = stab ~ g1 + g2 + g3 + g4 + tau1 + tau2 + tau3 + tau4  + p1 + p2 + p3, data = Stability_Data)
summary(mlr.model)
##Multiple Linear Regression using Lasso Regression Variables
mlr.model = lm(formula = stab ~ g1 + g2 + g3 + g4 + tau1 + tau2 + tau3 + tau4, data = Stability_Data)
summary(mlr.model)
##Quadratic Regression using Forward Selection Variables
quadratic.model = lm(formula = stab ~ poly(tau1, 2) + poly(tau2, 2) + poly(tau3, 2) + poly(tau4, 2) + poly(g1, 2) + poly(g2, 2) + poly(g3, 2) + poly(g4, 2) + poly(p3, 2), data = Stability_Data)
summary(quadratic.model)
##Quadratic Regression using Backward Elimination Variables
quadratic.model = lm(formula = stab ~ poly(tau1, 2) + poly(tau2, 2) + poly(tau3, 2) + poly(tau4, 2) + poly(g1, 2) + poly(g2, 2) + poly(g3, 2) + poly(g4, 2) + poly(p3, 2), data = Stability_Data)
summary(quadratic.model)
##Quadratic Regression using Stepwise Regression Variables
quadratic.model = lm(formula = stab ~ poly(tau1, 2) + poly(tau2, 2) + poly(tau3, 2) + poly(tau4, 2) + poly(g1, 2) + poly(g2, 2) + poly(g3, 2) + poly(g4, 2) + poly(p3, 2), data = Stability_Data)
summary(quadratic.model)
##Quadratic Regression using Ridge Regression Variables
quadratic.model = lm(formula = stab ~ poly(tau1, 2) + poly(tau2, 2) + poly(tau3, 2) + poly(tau4, 2) + poly(g1, 2) + poly(g2, 2) + poly(g3, 2) + poly(g4, 2) + poly(p1, 2) + poly(p2, 2) + poly(p3, 2), data = Stability_Data)
summary(quadratic.model)
##Quadratic Regression using Lasso Regression Variables
quadratic.model = lm(formula = stab ~ poly(tau1, 2) + poly(tau2, 2) + poly(tau3, 2) + poly(tau4, 2) + poly(g1, 2) + poly(g2, 2) + poly(g3, 2) + poly(g4, 2), data = Stability_Data)
summary(quadratic.model)
##Cubic Regression using Forward Selection Variables
cubic.model = lm(formula = stab ~ poly(tau1, 3) + poly(tau2, 3) + poly(tau3, 3) + poly(tau4, 3) + poly(g1, 3) + poly(g2, 3) + poly(g3, 3) + poly(g4, 3) + poly(p3, 3), data = Stability_Data)
summary(cubic.model)
##Cubic Regression using Backward Elimination Variables
cubic.model = lm(formula = stab ~ poly(tau1, 3) + poly(tau2, 3) + poly(tau3, 3) + poly(tau4, 3) + poly(g1, 3) + poly(g2, 3) + poly(g3, 3) + poly(g4, 3) + poly(p3, 3), data = Stability_Data)
summary(cubic.model)
##Cubic Regression using Stepwise Regression Variables
cubic.model = lm(formula = stab ~ poly(tau1, 3) + poly(tau2, 3) + poly(tau3, 3) + poly(tau4, 3) + poly(g1, 3) + poly(g2, 3) + poly(g3, 3) + poly(g4, 3) + poly(p3, 3), data = Stability_Data)
summary(cubic.model)
##Cubic Regression using Ridge Regression Variables
cubic.model = lm(formula = stab ~ poly(tau1, 3) + poly(tau2, 3) + poly(tau3, 3) + poly(tau4, 3) + poly(g1, 3) + poly(g2, 3) + poly(g3, 3) + poly(g4, 3) + poly(p1, 3) + poly(p2, 3) + poly(p3, 3), data = Stability_Data)
summary(cubic.model)
##Cubic Regression using Lasso Regression Variables
cubic.model = lm(formula = stab ~ poly(tau1, 3) + poly(tau2, 3) + poly(tau3, 3) + poly(tau4, 3) + poly(g1, 3) + poly(g2, 3) + poly(g3, 3) + poly(g4, 3), data = Stability_Data)
summary(cubic.model)