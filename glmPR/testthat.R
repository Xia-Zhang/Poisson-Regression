print("Please make sure that you have the glmPR package and put the PoissonExample.RData at current directory!")

library("glmPR")
setwd("E:/gsoc/")

load("PoissonExample.RData")
print("Load PoissonExample.Rdata... This data is from the glmnet webpage.")

print("First, test the glmPR related function")
glmPR::glmPR(x, y)
glmPR::glmPR(y ~ x, s = 1.0)
fit <- glmPR::glmPR(y ~ x)
predict(fit, x[1:5,])
summary(fit)


print("Then, compare the coefficient and predict ans with poisson in glm, which don't have the L1-norm penalization")
print("The glm:")
glmFit <- glm(y ~ x, family = "poisson")
print(glmFit)
df <- data.frame(x = I(x[1:5,]))
exp(predict(glmFit, df))
print("The glmPR:")
print(fit)
predict(fit, x[1:5,])

print("Finaly, compare the coefficient and predict ans with poisson in glm")
library(glmnet)
print("The glmnet:")
glmnetFit = glmnet::glmnet(x, y, family = "poisson")
coef(glmnetFit, s = 1)
exp(predict(glmnetFit, newx = x[1:5,], s = 1))
print("The glmPR:")
fit1 <- glmPR::glmPR(y ~ x, s = 1.0)
print(fit1)
predict(fit1, x[1:5,])
pritn("The observation is: 1, 7, 0, 1, 1")
print("I guess it is the difference of the optimaze way caused the difference. QAQ")
print("Glmnet uses an outer Newton loop, and an inner weighted least-squares loop (as in logistic regression) to optimize this criterion. 
	While in my package, I adopt the libLBFGS lib which use Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) method to optimize.
	")

