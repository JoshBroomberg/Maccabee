def extract_treat_and_control_data(covariates, treatment_status):
    X_treated = covariates[(treatment_status==1).to_numpy()]
    X_control = covariates[(treatment_status==0).to_numpy()]
    return X_treated, X_control
