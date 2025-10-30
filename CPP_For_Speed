#define ARMA_NO_DEBUG
#include <RcppArmadillo.h>
#include <stdlib.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// K_sob_cpp2, getK_sob, and getK_sob_prod use the same code from
// the GitHub page below for faster computation of Solobev Gram matrix
// https://github.com/jiayiwang1017/PCATE-balancing/blob/main/kernel_cpp.cpp
vec K_sob_cpp2(subview_col<double>  s, double t){
    vec k1s = s-0.5;
    double k1t = t-0.5;
    vec k1abst = abs(s-t)-0.5;

    vec k2s = (k1s % k1s - 1.0/12)/2;
    double k2t = (k1t * k1t - 1.0/12)/2;

    vec k4abst = (pow(k1abst,4) - k1abst%k1abst/2 + 7.0/240)/24;

    return 1.0 + k1s * k1t + k2s * k2t - k4abst;
}

mat getK_sob(const vec & t){
    int n = t.n_elem, i;
    mat out(n,n);
    for (i=0; i<n; i++){
        out.submat(i, i, n-1, i) = K_sob_cpp2(t.subvec(i, n-1), t[i]);
    }
    return symmatl(out);
}

// [[Rcpp::export]]
mat getK_sob_prod(const mat & X){
  mat K = ones<mat>(X.n_rows, X.n_rows);
  int i;
  for (i=0; i<X.n_cols; i++){
    K = K % getK_sob(X.col(i));
  }
  return K;
}

// K_gaussian_cpp2 and getGram_gauss are used for
// faster construction of Gaussian Gram matrix

// [[Rcpp::export]]
double K_gaussian_cpp2(const vec &s, const vec &t, double h) {
  vec temp = square(s - t);
  double cumul = sum(temp);  // Using Armadillo's sum function
  return exp(-0.5 * cumul / (h * h));
}

// [[Rcpp::export]]
mat getGram_gauss(const mat & designMat, double h){
    int N = designMat.n_rows;
    mat K = ones<mat>(N,N);
    for (int i=0; i<N; i++){
      for(int j=i+1; j<N; j++){
        K(i,j) = K_gaussian_cpp2(designMat.row(i).t(), designMat.row(j).t(), h);
        K(j,i) = K(i,j);
      }
    }
    return K;
}


// getGram_linear computes the Gram matrix for linear kernel

// [[Rcpp::export]]
mat getGram_linear(const mat &designMat) {
    return designMat * designMat.t(); // Compute Gram matrix directly
}


// del_k_x returns the gradient of the kernel function k_x()

// [[Rcpp::export]]
mat del_k_x(const vec &x1, const mat & designMat, double h){
  int N = designMat.n_rows;
  int p = designMat.n_cols;
  mat del = zeros<mat>(N,p);
  for(int i=0; i < N; i++){
    vec diff = x1-designMat.row(i).t();
    vec diff_sqr = square(diff);
    del.row(i) = diff / (h*h) * exp(-0.5 * sum(diff_sqr) / (h*h));
  }

  return del;
}

// hat_M_n and hat_M_n2 both are used to compute M_n(x) matrix
// hat_M_n2 takes in matrix G which is G = G_x^{-1} Gy  Gx^{-1}

// [[Rcpp::export]]
mat hat_M_n(const vec &x1, const mat & designMat, const mat & Gx, const mat & Gy, double eps_n, double h){
  int N = designMat.n_rows;
  int p = designMat.n_cols;
  mat del = zeros<mat>(N,p);
  for(int i=0; i < N; i++){
    vec diff = x1-designMat.row(i).t();
    vec diff_sqr = square(diff);
    del.row(i) = diff / (h*h) * exp(-0.5 * sum(diff_sqr) / (h*h));
  }

  mat Gx_eps = Gx + N * eps_n * eye(N,N);

  mat Gx_inv = inv(Gx_eps);

  mat hat_M_n_x = del.t() * Gx_inv * Gy * Gx_inv * del;

  return hat_M_n_x;
}

// [[Rcpp::export]]
mat hat_M_n2(const vec &x1, const mat & designMat, const mat & G, double h){
  int N = designMat.n_rows;
  int p = designMat.n_cols;
  mat del = zeros<mat>(N,p);
  for(int i=0; i < N; i++){
    vec diff = x1-designMat.row(i).t();
    vec diff_sqr = square(diff);
    //del.row(i) = diff / (h*h) * exp(-0.5 * sum(diff_sqr) / (h*h));
    del.row(i) = ((diff / (h*h)) * exp(-0.5 * sum(diff_sqr) / (h*h))).t();
  }

  mat hat_M_n_x = del.t() * G * del;

  return hat_M_n_x;
}

