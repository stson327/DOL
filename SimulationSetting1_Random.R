## Data generation code for randomized study in Seting 1
## We first generate the validation set with n_V = 10000
## Then we generate the training data

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


################### Train data generation

n <- 500; p <- 50 # n <- 1000

for(j in 1:100){
  
  Z <- replicate(p, runif(n,-2,2))
  X <- Z
  X[,1] <- exp(Z[,1]/2)
  X[,2] <- Z[,2] / (1 + exp(Z[,1]))
  X[,3] <- (Z[,1] * Z[,3] / 25 + 0.6)^3
  X[,4] <- (Z[,2] + Z[,4] + 20)^2
  
  V <- X %*% B0
  
  prop <- rep(1/2, n)
  
  A_binary <- rbinom(n, 1, prop)
  A <- 2*A_binary - 1
  pie <- rep(0, n)
  pie[A==1] <- prop[A==1]
  pie[A==-1] <- 1-prop[A==-1]
  
  ## True mean
  #mu <- Z[,1]*Z[,2]^3*Z[,3]^2*Z[,4]+
  mu <- 5 + 6*Z[,1]+8*Z[,2]+3*Z[,3]+5*Z[,4]+7*Z[,5]
  ate_val <-  5*sin(pi/((V[,1]+1)*sqrt(-V[,2]))) * V[,1] +
    2.5*(sin(pi * V[,1])) * log(-V[,2])
  
  Y <- mu + A*ate_val + rnorm(n, mean = 0, sd = 1)
  
  train <- data.frame(A, X, Y, Z=Z, A_binary, 
                      V1 = V[,1], V2 = V[,2])
  
}
