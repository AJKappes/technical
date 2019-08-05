### Gibbs Sampler ###

# Data
A <- c(62, 60, 63, 59)
B <- c(63, 67, 71, 64, 65, 66)
C <- c(68, 66, 71, 67, 68, 68)
D <- c(56, 62, 60, 61, 63, 64, 63, 59)
J <- 4
sigma <- var(A)

theta_update <- function(){
  theta_hat <- (mu/tau^2 + mean(A)/sigma^2)/(1/tau^2 + 1/sigma^2)
  var_theta <- 1/(1/tau^2 + 1/sigma^2)
  rnorm(J, theta_hat, sqrt(var_theta))
}

mu_update <- function(){
  rnorm(1, mean(theta), tau/sqrt(J))
}

tau_update <- function(){
  sqrt(1/rchisq(1, J-1) * sum((theta-mu)^2))
}

chains <- 5
iter <- 20
sims <- array(NA, c(iter, chains, J+2))
dimnames(sims) <- list(NULL, NULL,
                       c(paste('theta[', 1:4, ']', sep=''), 'mu', 'tau'))

for (m in 1:chains){
  mu <- rnorm(1, mean(A), sd(A))
  tau <- runif(1, 0, sd(A))
  for (t in 1:iter){
    theta <- theta_update()
    mu <- mu_update()
    tau <- tau_update()
    sims[t, m, ] <- c(theta, mu, tau)
  }
}

sims[,1,]
