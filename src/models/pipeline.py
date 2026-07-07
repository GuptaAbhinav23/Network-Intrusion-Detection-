from .machine_learning import logistic_regression, decision_tree, random_forest, extra_trees, adaboost, xgboost, lightgbm, catboost, linear_svm, knn
 
def main():

    logistic_regression.train()
    decision_tree.train()
    random_forest.train()
    extra_trees.train()
    adaboost.train()
    xgboost.train()
    lightgbm.train()
    catboost.train()
    linear_svm.train()
    # knn.train()
    
    

if __name__ == "__main__":
    main()