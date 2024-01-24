data {
  int<lower=0> N;          // Number of data points
  vector[N] y;             // Data
  real mu_prior_mean;      // Prior mean for the Gaussian prior
  real<lower=0> mu_prior_sd;  // Prior standard deviation
  real<lower=0> sigma;     // Known standard deviation of the Gaussian likelihood
}

parameters {
  real mu;  // Mean to be estimated
}

model {
  // Prior for 'mu'
  mu ~ normal(mu_prior_mean, mu_prior_sd);

  // Likelihood of the data
  y ~ normal(mu, sigma);
}
