library(MASS)
library(dplyr)
library(model4you)
library(grf)
library(mboost)
library(glmnet)
library(pROC)

#setwd("~/deep learning/R")
rm(list=ls())

# Data generation function
data.gen <- function(n, k, prevalence = sqrt(0.5), prog.eff = 1, sig2, y.sig2, rho, rhos.bt.real, a.constent) {
  covm <- matrix(rho * sig2, k, k)
  diag(covm) <- sig2
  covm[1:2, 3:k] <- rhos.bt.real
  covm[3:k, 1:2] <- rhos.bt.real
  dich.cutoff <- qnorm(prevalence, sd = sqrt(sig2))
  
  x <- mvrnorm(n, rep(0, k), covm)
  x.dich <- 1 * (x < dich.cutoff)
  w <- x.dich[, 3:5]
  trt <- rbinom(n, 1, prob = 0.5)
  
  prog.part <- prog.eff * rowSums(w) + rnorm(n, sd = sqrt(y.sig2))
  pred.part <- a.constent * (-1 * prevalence + (x[, 1] * x[, 2]) < dich.cutoff)
  
  y.0 <- prog.part
  y.1 <- pred.part + prog.part
  
  y <- trt * y.1 + (1 - trt) * y.0
  y.binary <- 1 * (plogis(trt * y.1 + (1 - trt) * y.0) > prevalence)
  
  surv.time.0 <- exp(y.0)
  surv.time.1 <- exp(y.1)
  cens.time <- exp(rnorm(n, sd = 3))
  
  y.time.to.event <- pmin(trt * surv.time.1 + (1 - trt) * surv.time.0, cens.time)
  status <- 1 * (trt * surv.time.1 + (1 - trt) * surv.time.0 <= cens.time)
  
  sigs <- 1 * ((x[, 1] * x[, 2]) < dich.cutoff)
  
  data <- data.frame(y = y, y.binary = y.binary, y.time.to.event = y.time.to.event, status = status,
                     y.0 = y.0, y.1 = y.1, treatment = trt, sigpos = sigs, x)
  
  return(list(data = data, sigpos = sigs))
}

# Simulation parameters
n <- 1000
k <- 10
prevalence <- sqrt(0.5)
rho <- 0.2
sig2 <- 2
rhos.bt.real <- rep(0.1 * sig2, k - 2)
y.sig2 <- 1
simulation_num <- 1000
prog.eff.list <- c(0, 1, 2)
effect.size.list <- c(0, 0.1, 0.5, 0.7, 1, 2, 3)

# Run the simulation
run_simulation <- function(prog.eff, effect.size) {
  results <- list(causal_df = NULL, linear_df = NULL, importance_df = NULL, importance_df_lm = NULL)
  
  for (j in 1:simulation_num) {
    set.seed(j)
    
    # Generate data
    a.constent <- effect.size / (2 * (1 - prevalence))
    ObsData <- data.gen(n = n, k = k, prevalence = prevalence, prog.eff = prog.eff, sig2 = sig2, 
                        y.sig2 = y.sig2, rho = rho, rhos.bt.real = rhos.bt.real, a.constent = a.constent)
    savedata <- ObsData$data
    
    train_idx <- sample(1:n, n * 0.8)
    train_data <- savedata[train_idx, ]
    test_data <- savedata[-train_idx, ]
    
    y_train <- train_data$y
    x_train <- train_data[, grep("^x", names(train_data))]
    
    # Method 1: Causal Forest
    tau.forest <- causal_forest(X = x_train, Y = y_train, W = train_data$treatment)
    c.pred_test <- predict(tau.forest, test_data[, grep("^x", names(test_data))])
    c.pred_train <- predict(tau.forest, x_train)
    
    auc_train <- auc(train_data$sigpos, 1.0 / (1 + exp(-c.pred_train$predictions)))
    auc_test <- auc(test_data$sigpos, 1.0 / (1 + exp(-c.pred_test$predictions)))
    results$causal_df <- rbind(results$causal_df, c(auc_train, auc_test))
    
    grf.vmp <- variable_importance(tau.forest)
    importance_causal <- order(grf.vmp, decreasing = TRUE)[1:2]
    results$importance_df <- rbind(results$importance_df, importance_causal)
    
    # Method 2: Linear Regression with modified covariates
    y_train_lm <- 2 * (2 * train_data$treatment - 1) * y_train
    lm_mod <- lm(y_train_lm ~ ., data = train_data[, -1])
    
    c.pred_test <- predict(lm_mod, test_data[, grep("^x", names(test_data))])
    c.pred_train <- predict(lm_mod, x_train)
    
    auc_train_lm <- auc(train_data$sigpos, 1.0 / (1 + exp(-c.pred_train)))
    auc_test_lm <- auc(test_data$sigpos, 1.0 / (1 + exp(-c.pred_test)))
    results$linear_df <- rbind(results$linear_df, c(auc_train_lm, auc_test_lm))
    
    lm_importance <- abs(coef(lm_mod)[-1])
    importance_linear <- order(lm_importance, decreasing = TRUE)[1:2]
    results$importance_df_lm <- rbind(results$importance_df_lm, importance_linear)
  }
  
  return(results)
}

# Perform the simulation and save results
for (pr in prog.eff.list) {
  for (pe in effect.size.list) {
    result <- run_simulation(pr, pe)
    result_auc <- data.frame(cbind(result$causal_df, result$linear_df))
    colnames(result_auc) <- c('cau_train', 'cau_test', 'lm_train', 'lm_test')
    mean_row <- colMeans(result_auc)
    result_auc <- rbind(result_auc, mean_row)
    # Save variable importance for causal and linear models
    feature_table_df <- data.frame(table(cbind(result$importance_df[, 1], result$importance_df[, 2])))
    feature_table_df_lm <- data.frame(table(cbind(result$importance_df_lm[, 1], result$importance_df_lm[, 2])))
    
  }
}
