library(causalToolbox)


HRF_learners <- function(X_train, y_train, t_train, X_test, tau_test, nthreads=0) {
  mu_forest_params <- list(
    relevant.Variable = 1:ncol(X_train),
    ntree = 50,
    replace = TRUE,
    sample.fraction = 0.9,
    mtry = ncol(X_train),
    nodesizeSpl = 1,
    nodesizeAvg = 3,
    splitratio = .5,
    middleSplit = FALSE)

  tau_forest_params <- list(
    relevant.Variable = 1:ncol(X_train),
    ntree = 50,
    replace = TRUE,
    sample.fraction = 0.7,
    mtry = round(ncol(X_train) * 17 / 20),
    nodesizeSpl = 5,
    nodesizeAvg = 6,
    splitratio = 0.8,
    middleSplit = TRUE)

  e_forest_params <- list(
    relevant.Variable = 1:ncol(X_train),
    ntree = 25,
    replace = TRUE,
    sample.fraction =  0.5,
    mtry = ncol(X_train),
    nodesizeSpl = 11,
    nodesizeAvg = 33,
    splitratio = .5,
    middleSplit = FALSE)

  # T learner
  tl_rf <- T_RF(
    feat=X_train, tr=t_train, yobs=y_train,
    nthread=nthreads,
    mu0.forestry=mu_forest_params,
    mu1.forestry=mu_forest_params)

  T_cate_rf <- EstimateCate(tl_rf, X_test)
  T_loss <- mean((T_cate_rf - tau_test) ^ 2)
  remove(tl_rf)

  # S learner
  sl_rf <- S_RF(feat=X_train, tr=t_train, yobs=y_train,
    nthread=nthreads,
    mu.forestry=mu_forest_params)
  S_cate_rf <- EstimateCate(sl_rf, X_test)
  S_loss <- mean((S_cate_rf - tau_test) ^ 2)
  remove(sl_rf)

  # X learner
  xl_rf = X_RF(feat=X_train, tr=t_train, yobs=y_train,
    nthread=nthreads,
    mu.forestry=mu_forest_params,
    tau.forestry=tau_forest_params,
    e.forestry=e_forest_params)

  X_cate_rf = EstimateCate(xl_rf, X_test)
  X_loss = mean((X_cate_rf - tau_test) ^ 2)
  remove(xl_rf)

  c(T_loss, S_loss, X_loss)
}


# create example data set
#simulated_experiment <- simulate_causal_experiment(
#    ntrain = 1000,
#    ntest = 100,
#    dim = 10)

#feature_train <- simulated_experiment$feat_tr
#w_train <- simulated_experiment$W_tr
#yobs_train <- simulated_experiment$Yobs_tr
#cate_true <- simulated_experiment$tau_tr

#print(HRF_learners(cate_true, feature_train, yobs_train, w_train))
