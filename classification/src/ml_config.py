svm_parameter_list = [
    {
        "C":c,
        "tol":tol,
        "loss":loss,
    }
    for c in [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
    for loss in ["hinge", "squared_hinge"]
    for tol in [1e-3, 1e-4, 1e-5, 1e-6]
]

rf_parameter_list = [
    {
        "n_estimators":n,
        "max_depth":max_depth,
        "criterion":c,
    }
    for n in [50, 100, 150, 200, 250, 300]
    for max_depth in [50, 100, 150, 200, 250, 300, None]
    for c in ["gini", "entropy"]
]
rf_parameter_list = [{"n_estimators":150, "max_depth":None, "criterion":"gini"}]

parameter_dict = {
    "SVM":svm_parameter_list,
    "RandomForest":rf_parameter_list,
}
