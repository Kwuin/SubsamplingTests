data {
  int<lower=0> N; // Number of observations
  int<lower=0, upper=1> y[N]; // Binary outcome variable
  int<lower=1> K; // Number of predictors
  matrix[N, K] X; // Predictor matrix (includes a column of 1s if you're including an intercept)
}

parameters {
  vector[K] beta; // Regression coefficients (including intercept as beta[1])
}

model {
  // Priors for regression coefficients
  beta ~ normal(0, 5);
  
  // Logistic regression model
  y ~ bernoulli_logit(X * beta);
}
