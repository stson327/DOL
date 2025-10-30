
library(splines)
library(nprobust)
library(np)
library(KernSmooth)
library(DTRlearn2)
library(rlearner)
library(KRLS2)

Rcpp::sourceCpp("CPP_For_Speed.cpp")
source("FunctionSet.R")

################################ Generation of test/validation set 


p <- 50; n_V <- 10000

### Validation set generation

## Latent varialbes from uniform distribution
Z_test <- replicate(p, runif(n_V, -2,2))

## True projection to the central mean space
B0 <- matrix(0, nr = p, nc = 2)
a = sqrt(0.6^2 + 0.5^2 + (-0.2)^2)
b = sqrt(        0.2^2 + 0.5^2 + (-0.5)^2)

B0[c(1,2,3),1] <- c(0.6, 0.5, -0.2) / a
B0[c(2,3,4),2] <- c(     0.2, 0.5, -0.5) / b 


## Covariates defined by latent variables
X_test <- Z_test
X_test[,1] <- exp(Z_test[,1]/2)
X_test[,2] <- Z_test[,2] / (1 + exp(Z_test[,1]))
X_test[,3] <- (Z_test[,1] * Z_test[,3] / 25 + 0.6)^3
X_test[,4] <- (Z_test[,2] + Z_test[,4] + 20)^2

V_test <- X_test %*% B0

## Probability if treatment assignment -- 1/2 for the randomized study
prop_V <- rep(1/2, n_V)

A_binary <- rbinom(n_V, 1, prop_V)
A_V <- 2*A_binary - 1

## True mean
Z <- Z_test
V <- V_test

## Main effect
mu <- 5 + 6*Z[,1]+8*Z[,2]+3*Z[,3]+5*Z[,4]+7*Z[,5]

## Heterogeneous treatment effect
ate_val <-  5*sin(pi/((V[,1]+1)*sqrt(-V[,2]))) * V[,1] +
  2.5*(sin(pi * V[,1])) * log(-V[,2])
hist(ate_val, breaks = 30)

## Reward from normal distribution
Y_test <- mu + A_V*ate_val + rnorm(n_V, mean = 0, sd = 1)
test <- data.frame(A = A_V, X = X_test, Y = Y_test)

## Optimal decision: Non-linear decision rule
d_star <- ate_val > 0
d_star <- d_star * 2 - 1
sum(d_star)
## pi(A,X) = pie
pie_V <- rep(0, n_V)
pie_V[A_V==1] <- prop_V[A_V==1]
pie_V[A_V==-1] <- 1-prop_V[A_V==-1]

## Optimal reward based on the optimal decision rule d_star, the Bayes decision rule
optimal <- mean(Y_test * (A_V == d_star) / pie_V) / mean((A_V == d_star) / pie_V)

###############################


################################ Generation of train set

n <- 1000; p <- 50

## True projection to the central mean space

B0 <- matrix(0, nr = p, nc = 2)

a = sqrt(0.6^2 + 0.5^2 + (-0.2)^2)
b = sqrt(        0.2^2 + 0.5^2 + (-0.5)^2)

B0[c(1,2,3),1] <- c(0.6, 0.5, -0.2) / a
B0[c(2,3,4),2] <- c(     0.2, 0.5, -0.5) / b 

## Generation of features

Z <- replicate(p, runif(n,-2,2))
X <- Z
X[,1] <- exp(Z[,1]/2)
X[,2] <- Z[,2] / (1 + exp(Z[,1]))
X[,3] <- (Z[,1] * Z[,3] / 25 + 0.6)^3
X[,4] <- (Z[,2] + Z[,4] + 20)^2

V <- X %*% B0


##Propensity score -- random
prop <- rep(1/2, n)

A_binary <- rbinom(n, 1, prop)
A <- 2*A_binary - 1
pie <- rep(0, n)
pie[A==1] <- prop[A==1]
pie[A==-1] <- 1-prop[A==-1]

## Main effect
mu <- 5 + 6*Z[,1]+8*Z[,2]+3*Z[,3]+5*Z[,4]+7*Z[,5]

## Heterogeneous treatment effect
ate_val <-  5*sin(pi/((V[,1]+1)*sqrt(-V[,2]))) * V[,1] +
  2.5*(sin(pi * V[,1])) * log(-V[,2])


## Generation of response
Y <- mu + A*ate_val + rnorm(n, mean = 0, sd = 1)

train <- data.frame(A, X, Y, Z=Z, A_binary, 
                    V1 = V[,1], V2 = V[,2])


################################ Learning the decision rule

assignT <- as.numeric(A == +1) ## 1 if A = +1 nd 0 if A = -1
weight <- 1/pie

prop <- 1/2

gx <- predict(lm((weight-1)*Y ~ X)) #estimate g(x)
Yy <- assignT * (Y-gx) * weight - (1-assignT) * (Y-gx) * weight #pseudo-outcome

h <- c(median_heuristic(X), 1)

d <- gKDR_CV_d_w(X=X, Yy=Yy, y0=Y, y=Y, A=A, candi_d=1:5, foldsize=2,
                 cost = seq(0.05, 2, length.out=20), h1=h[1], h2=h[2], 
                 weight=weight, eps=1e-7, gx = gx, aol=TRUE)

result_v_aol_ghat[j,4]<- d <- d[1]

B <- gKDR_manual_linear(X=X,Yy=Yy,d=d,h=h[1],eps=1e-7)

B_hat <- B[[1]]

B_hat <- as.matrix(B_hat)
V_hat <- X %*% as.matrix(B_hat)

aa <- B_hat %*% solve(t(B_hat) %*% B_hat) %*% t(B_hat)
bb <- B0 %*% t(B0)

result_v_aol_ghat[j,3] <- sqrt(sum((aa-bb)^2)) 

V_hat <- data.frame(V_hat);
names(V_hat) <- paste("V", 1:d, "_hat", sep = "")

train <- data.frame(train, V_hat)

h1_V_hat <- median_heuristic(as.matrix(V_hat))
covarV <- names(V_hat)

costV <- cv.cost_w(foldsize=2, n=nrow(train), cost=seq(0.05, 2, length.out=20),
                   data=train, covar=covarV, weight=weight, h1 = h1_V_hat,
                   gx=gx, aol=TRUE)

## Fit decision rule
f_wsvm_v <- wsvm_solve(X=V_hat, A=A*sign(Y-gx), 
                       wR=as.matrix(abs(Y-gx)*weight), kernel='rbf', 
                       sigma=1/(2*h1_V_hat^2), C = costV, e=1e-8)

## Retrieve coefficients
v_est <- f_wsvm_v$alpha1
b_est <- f_wsvm_v$beta0
#################################


################################ Evaluation of decision rule 

V_hat_test <- X_test %*% as.matrix(B_hat)
## Decision rule values
f_hat_v <- apply(V_hat_test, 1, function(x){
  f_fit(trainSize=nrow(train), x=x, 
        trainX=V_hat, h1=h1_V_hat, 
        beta=v_est, beta0=b_est)
} 
)

## Decisions using the test set by using the estimated rule on the test data
d_v <- sign(f_hat_v)

## Accuracy
acc_v <- sum(d_v == d_star) / n_V
## Value function: Expected reward
val_v <- mean(Y_test*(A_V == d_v)/pie_V)/mean((A_V == d_v) / pie_V)




