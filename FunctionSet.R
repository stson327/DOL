
## Computation for median heuristic bandwidth for kernel functions

median_heuristic <- function(X){
  
  X <- as.matrix(X)
  n <- nrow(X); p <- ncol(X)
  
  if(p > 1){
    X_ij_sqr_dist <- matrix(0,nr=n,nc=n)
    for (i in (1:n)){
      for (j in (1:i)){
        X_ij_sqr_dist[i,j] <- sum((X[i,] - X[j,])^2)
      }
    }
    h1 <- median(X_ij_sqr_dist[lower.tri(X_ij_sqr_dist)])
    h1 <- sqrt(h1)
  }else{
    X_ij_sqr_dist <- matrix(0,nr=n,nc=n)
    for (i in (1:n)){
      for (j in (1:i)){
        X_ij_sqr_dist[i,j] <- sum((X[i] - X[j])^2)
      }
    }
    h1 <- median(X_ij_sqr_dist[lower.tri(X_ij_sqr_dist)])
    h1 <- sqrt(h1)
  }
  
  return(h1)
  
}


## Tuning for solving the regularization parameter in the objective function
## for AOL

cv.cost_w <- function(foldsize, n, cost=seq(0.01, 2, length.out=10),
                      data, covar, weight, h1, gx = 0, aol= FALSE){
  n_star <- n - n %% foldsize
  test_size <- n_star/foldsize
  train_size <- n - test_size
  
  k_fold <- numeric(foldsize)
  avg_val_cost <- numeric(length(cost))
  
  for(c_ind in 1:length(cost)){
    for(fold in 1:foldsize){
      test_index <- (test_size*fold-test_size + 1):(test_size*fold)
      tst <- data[test_index,]; weight_tst <- weight[test_index]
      trn <- data[-test_index,]; weight_trn <- weight[-test_index]
      X <- data[,covar] ; X <- as.matrix(X)
      trnX <- as.matrix(X[-test_index,])
      tstX <- as.matrix(X[test_index,])
      
      if(aol == TRUE){
        tst_gx <- gx[test_index]
        trn_gx <- gx[-test_index]
      }else{
        tst_gx <- 0
        trn_gx <- 0
      }
      
      
      if(aol == FALSE){
        f_wsvm <- wsvm_solve(X=trnX, A=trn$A, wR=as.matrix(trn$Y*weight_trn),
                             kernel='rbf',
                             sigma=1/(2*h1^2), C = cost[c_ind], e=1e-8)
      }else{
        f_wsvm <- wsvm_solve(X=trnX, A=trn$A*sign(trn$Y-trn_gx),
                             wR=as.matrix(abs(trn$Y-trn_gx)*weight_trn),
                             kernel='rbf',
                             sigma=1/(2*h1^2), C = cost[c_ind], e=1e-8)
      }
      
      v_est <- f_wsvm$alpha1
      b_est <- f_wsvm$beta0
      
      f_hat <- apply(tstX, 1, function(x){
        f_fit(trainSize=nrow(trnX), x=x,
              trainX=trnX, h1=h1,
              beta=v_est, beta0=b_est)
      }
      )
      d_x <- sign(f_hat)
      k_fold[fold] <- mean(tst$Y*(tst$A==d_x)*weight_tst)/
        mean((tst$A==d_x)*weight_tst)
    }
    avg_val_cost[c_ind] <- mean(k_fold, na.rm = TRUE)
  }
  #plot(cost, avg_val_cost)
  cost<- cost[avg_val_cost == max(avg_val_cost)]
  cost <- min(cost)
  
  return(cost)
  
  
}

## k-fold cross-validation implementation to choose dimension for gKDR

gKDR_CV_d_w <- function(X, Yy, y0, y, A, candi_d, foldsize,
                        cost, h1, h2, weight, eps, gx=0, aol=FALSE){
  
  Yy <- as.matrix(Yy); X <- as.matrix(X)
  n <- nrow(X); p <- ncol(X)
  
  n_star <- n - n %% foldsize
  
  #val_est_d <- numeric(length(candi_d))
  val_est_d <- matrix(0, nr = foldsize, nc = length(candi_d))
  
  for(k in 1:foldsize){
    
    test_index <- ((k-1)*n_star/foldsize+1):(k*n_star/foldsize)
    
    weight_tst <- weight[test_index]; weight_trn <- weight[-test_index]
    
    tstX <- X[test_index,]; trnX <- X[-test_index,]
    
    tstYy <- Yy[test_index,]; tst_Y <- y[test_index]
    trnYy <- Yy[-test_index,]; trn_Y <- y[-test_index]
    
    tst_Y0 <- y0[test_index]; trn_Y0 <- y0[-test_index]
    
    tstA <- A[test_index]; trnA <- A[-test_index]
    
    trn <- data.frame(A = trnA, trnX, Y = trn_Y)
    
    if(aol == TRUE){
      tst_gx <- gx[test_index]
      trn_gx <- gx[-test_index]
    }else{
      tst_gx <- 0
      trn_gx <- 0
    }
    
    n_test <- length(test_index)
    n_train <- n - length(test_index)
    
    Gx <- getGram_gauss(designMat = as.matrix(trnX), h = h1)
    Gy <- getGram_linear(designMat = as.matrix(trnYy))
    tilde_M <- array(NA, dim = c(p, p, n_train))
    for(i in 1:n_train){
      M_est_trn <- hat_M_n(x1 = trnX[i,], designMat = trnX,
                           Gx = Gx, Gy = Gy, eps_n = 1e-4,
                           h = h1)
      tilde_M[,,i] <- as.matrix(M_est_trn)
    }
    
    tilde_M <- apply(tilde_M, 1:2, mean)
    
    decompose <- eigen(tilde_M)
    eigs <- decompose$values
    vecs <- decompose$vectors
    
    trn0 <- trn
    for(cd in 1:length(candi_d)){
      trn <- trn0
      d <- candi_d[cd]
      B_hat <- vecs[,(1:d)]
      
      V_hat <- trnX %*% as.matrix(B_hat)
      
      V_hat <- data.frame(V_hat); B_hat <- data.frame(B_hat)
      names(V_hat) <- paste("V", 1:d, "_hat", sep = "")
      #names(B_hat) <- paste("B", 1:d, "_est", sep = "")
      trn <- data.frame(trn, V_hat)#, B_hat)
      
      h1_V_hat <- median_heuristic(as.matrix(V_hat))
      covarV <- names(V_hat)
      
      costV<- cv.cost_w(foldsize=foldsize, n=nrow(trn), cost=cost,
                        data=trn, covar=covarV, weight=weight_trn, h1=h1_V_hat,
                        gx = trn_gx, aol = aol)
      #f_wsvm_v <- wsvm_solve(X=V_hat, A=trnA, wR=trn_Y * weight, kernel='rbf',
      #                       sigma=1/(2*h1_V_hat^2), C = costV, e=1e-8)
      
      if(aol == FALSE){
        f_wsvm_v <- wsvm_solve(X=V_hat, A=trnA,
                               wR=as.matrix(trn_Y*weight_trn), kernel='rbf',
                               sigma=1/(2*h1_V_hat^2), C = costV, e=1e-8)
      }else{
        f_wsvm_v <- wsvm_solve(X=V_hat, A=trnA*sign(trn_Y-trn_gx),
                               wR=as.matrix(abs(trn_Y-trn_gx)*weight_trn),
                               kernel='rbf',
                               sigma=1/(2*h1_V_hat^2), C = costV, e=1e-8)
      }
      
      v_est <- f_wsvm_v$alpha1
      b_est <- f_wsvm_v$beta0
      ####################################################################
      
      V_hat_test <- tstX %*% as.matrix(B_hat)
      f_hat_v <- apply(V_hat_test, 1, function(x){
        f_fit(trainSize=nrow(trn), x=x,
              trainX=V_hat, h1=h1_V_hat,
              beta=v_est, beta0=b_est)
      }
      )
      d_v <- sign(f_hat_v)
      
      val_est_d[k,cd] <- mean(tst_Y0*(tstA == d_v) * weight_tst)/
        mean((tstA == d_v) * weight_tst)
    }
    
  }
  
  val_est_d <- apply(val_est_d, 2, mean)
  
  best <- candi_d[val_est_d == max(val_est_d)]
  
  return(best)
  
}

## AOL objective function solver that uses quadratic programming
## The code is slight variation of wsmv_solve in DTRlearn2 pakcage
## by  Yuan Chen

wsvm_solve <-function(X, A, wR, kernel='linear', sigma=0.05, C=1, e=1e-7) {
  
  if (kernel=='linear') {
    K = X %*% t(X)
    if (is.vector(X)) K = t(X) %*% X
  }else if (kernel=='rbf'){
    rbf = rbfdot(sigma = sigma)
    K = kernelMatrix(rbf, as.matrix(X))
  }
  
  y = A
  H = y %*% t(y) * K
  H = H + 1e-8 * diag(NCOL(K)) %*% (tcrossprod(wR))
  
  
  n = length(A)
  solution <- tryCatch(ipop(c = rep(-1, n), H = H, A = t(y), b = 0,
                            l = numeric(n), u = C*wR, r = 0),
                       error=function(er) er)
  if ("error" %in% class(solution)) {
    return(list(beta0=NA, beta=NA, fit=NA, probability=NA,
                treatment=NA, sigma=NA, H=NA, alpha1=NA))
  }
  alpha = primal(solution)
  alpha1 = alpha * y
  
  if (kernel=='linear'){
    w = t(X) %*% alpha1
    fitted = X %*% w
  } else if (kernel=='rbf'){
    fitted = K %*% alpha1
  }
  rm = y - fitted
  Imid = (alpha < C-e) & (alpha > e)
  rmid = rm[Imid==1]
  if (sum(Imid)>0){
    bias = mean(rmid)
  } else {
    Iup = ((alpha<e)&(A==-sign(wR)))|((alpha>C-e)&(A==sign(wR)))
    Ilow = ((alpha<e)&(A==sign(wR)))|((alpha>C-e)&(A==-sign(wR)))
    rup = rm[Iup]
    rlow = rm[Ilow]
    bias = (min(rup)+max(rlow))/2
  }
  fit = bias + fitted
  prob = exp(fit) / (1+ exp(fit))
  
  
  if (kernel=='linear') {
    model = list(beta0=bias, beta=w, fit=fit, probability=prob,
                 treatment=2*(fit>0)-1, alpha1=alpha1) #, solution=solution)
    class(model)<-'linearcl'
  } else if (kernel=='rbf') {
    model = list(beta0=bias, fit=fit, probability=prob,
                 treatment=2*(fit>0)-1, sigma=sigma, H=X, alpha1=alpha1)
    class(model) = 'rbfcl'
  }
  return (model)
}

## Retrieving decision function values evaluated at given data points
## by using the fitted decision function
f_fit <- function(trainSize, x, trainX, h1, beta, beta0){
  testX <- rep(x,trainSize)
  testX <- matrix(testX, trainSize, byrow = TRUE)
  f_hat_x <- sum(beta*exp(-0.5*apply((testX-trainX)^2,1,sum)/h1^2)) +
    beta0
  return(f_hat_x)
}



# Hinge loss function
hinge <- function(x){
  return(max(x, 0))
}


## Computation for the gKDR projection matrix B
## Manual input of d 

gKDR_manual_linear <- function(X, Yy, d, h, eps){
  
  ## X: Covariates
  ## Yy: outcome
  ## d: chosen dimension
  ## h: bandwidth for Gaussian kernel used for the covariates
  ## eps: Tikhonov regulaization
  
  Yy <- as.matrix(Yy)
  n <- nrow(X); p <- ncol(X)
  
  Gx_eps <- getGram_gauss(designMat = as.matrix(X), h = h) + n * eps * diag(n)
  Gy <- getGram_linear(designMat = Yy)
  
  Gx_inv <- Matrix::solve(Gx_eps)
  G <- Gx_inv %*% Gy %*% Gx_inv
  
  tilde_M <- matrix(0,p,p)
  for(i in 1:n){
    tilde_M <- tilde_M + hat_M_n2(x1=as.numeric(X[i,]),designMat=as.matrix(X),
                                  G = G, h = h)
  }
  tilde_M <- tilde_M/n
  
  decompose <- eigen(tilde_M)
  eigs <- decompose$values
  vecs <- decompose$vectors
  
  B <- vecs[,(1:d)]
  info_prop <- sum(eigs[1:d]) / sum(eigs)
  
  return(list(B, info_prop))
  
}

## R-learner using Gaussian kernel ridge regression
## that uses reduced covariates from gKDR
rkern_gKDR = function(x, w, y,
                      k_folds = NULL,
                      p_hat = NULL,
                      m_hat = NULL,
                      b_rangeX = 10^(seq(-3,3,0.5)),
                      b_range = 10^(seq(-3,3,0.5)),
                      lambda_range = 10^(seq(-3,3,0.5)),
                      B = B){
  
  v = x %*% B
  
  if (is.null(k_folds)) {
    k_folds = floor(max(3, min(10,length(w)/4)))
  }
  w = as.numeric(w)
  
  if (is.null(p_hat)) {
    p_hat_model = cv_klrs(x, w, weights=NULL, k_folds=k_folds, 
                          b_range=b_rangeX,lambda_range=lambda_range)
    p_hat = p_hat_model$fit
  } else {
    p_hat_model = NULL
  }
  if (is.null(m_hat)) {
    m_hat_model = cv_klrs(x, y, weights=NULL, k_folds=k_folds, 
                          b_range=b_rangeX,lambda_range=lambda_range)
    m_hat = m_hat_model$fit
  } else {
    m_hat_model = NULL
  }
  
  # lambda is the same as var from kernlab. 1/b is the same as sigma in kernlab
  model_tau_cv= cv_klrs(v, (y-m_hat)/(w-p_hat), weights = (w-p_hat)^2, 
                        k_folds=k_folds, b_range=b_range,
                        lambda_range=lambda_range)
  
  ret = list(tau_fit = model_tau_cv)
  class(ret) <- "rkern_gKDR"
  ret
}
