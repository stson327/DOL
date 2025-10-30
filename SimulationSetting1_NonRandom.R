## Data generation code for non-randomized study in Seting 1
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
V <- V_test
Z <- Z_test

## Propensity score
Z_tilde <- apply(Z[,c(2,4,6,8,10)], 1, sum) / 5
prop_V <- exp( -Z[,1] - 5 * Z_tilde ) / 
  (1 + exp( -Z[,1] - 5 * Z_tilde ))

A_V <- rbinom(n_V, 1, prop_V)
A_V <- 2*A_V - 1 

## Main effect
mu <- 5 + 6*Z[,1]+8*Z[,2]+3*Z[,3]+5*Z[,4]+7*Z[,5]
## Heterogeneous treatment effect
ate_val <-  5*sin(pi/((V[,1]+1)*sqrt(-V[,2]))) * V[,1] +
            2.5*(sin(pi * V[,1])) * log(-V[,2])
hist(ate_val, breaks = 30)


d_star <- ate_val > 0
d_star <- d_star * 2 - 1

Q0_V <- mu + A_V * ( ate_val )

## Reward from normal distribution
Y_test <- Q0_V + rnorm(n_V, mean = 0, sd = 1)
test <- data.frame(A = A_V, X = X_test, Y = Y_test)

## Optimal decision: Non-linear decision rule
## Probability of beig assigned to +1 group
pie_V <- rep(0, n_V)
pie_V[A_V==1] <- prop_V[A_V==1]
pie_V[A_V==-1] <- 1-prop_V[A_V==-1]

## Optimal reward based on the optimal decision rule d_star
optimal <- mean(Y_test * (A_V == d_star) / pie_V) / mean((A_V == d_star) / pie_V)

################### Train data

n <- 500; p <- 50 # n <- 1000

Z <- replicate(p, runif(n,-2,2))
X <- Z
X[,1] <- exp(Z[,1]/2)
X[,2] <- Z[,2] / (1 + exp(Z[,1]))
X[,3] <- (Z[,1] * Z[,3] / 25 + 0.6)^3
X[,4] <- (Z[,2] + Z[,4] + 20)^2

V <- X %*% B0

Z_tilde <- apply(Z[,c(2,4,6,8,10)], 1, sum) / 5

prop <- exp( -Z[,1] - 5 * Z_tilde ) / 
  (1 + exp( -Z[,1] - 5 * Z_tilde ))

A_binary <- rbinom(n, 1, prop)
A <- 2*A_binary - 1
pie <- rep(0, n)
pie[A==1] <- prop[A==1]
pie[A==-1] <- 1-prop[A==-1]

mu <- 5 + 6*Z[,1]+8*Z[,2]+3*Z[,3]+5*Z[,4]+7*Z[,5]
ate_val <-  5*sin(pi/((V[,1]+1)*sqrt(-V[,2]))) * V[,1] +
  2.5*(sin(pi * V[,1])) * log(-V[,2])

Q0 <- mu + A * ( ate_val )

Y <- Q0 + rnorm(n, mean = 0, sd = 1)

## Estimate w_hat

OUTstd <- transform.sob(X)
Xstd <- OUTstd$Xstd
Xlim <- OUTstd$Xlim
K <- getK_sob_prod(as.matrix(Xstd))

# design a grid for the tuning parameter
nlam <- 50
lams <- exp(seq(log(1e-6), log(1), len=nlam))

# compute weights for T=1
fit1 <- ATE.ncb.SN(A_binary, K, lam1s=lams)
# compute weights for T=0
fit0 <- ATE.ncb.SN(1-A_binary, K, lam1s=lams)

wgt <- fit1$w + fit0$w

logist <- glm(A_binary ~ X, family = "binomial")
prop <- predict(logist, type = "response")

pie <- rep(0, n)
pie[A==1] <- prop[A==1]
pie[A==-1] <- 1-prop[A==-1]

logist.cv <- cv.glmnet(X, A_binary, family = "binomial", 
                       lambda = seq(0.0001, 0.1, length.out = 100))
logist <- glmnet(X, A_binary, family = "binomial", 
                 lambda = logist.cv$lambda.min)

prop <- predict(logist, type = "response", newx = X)
pie_lasso <- rep(0, n)
pie_lasso[A==1] <- prop[A==1]
pie_lasso[A==-1] <- 1-prop[A==-1]

train <- data.frame(A, X, Y, Z = Z, A_binary, Q0,
                    V1=V[,1], V2=V[,2],
                    wgt, inv_logit = 1/pie, inv_logit_lasso = 1/pie_lasso)

