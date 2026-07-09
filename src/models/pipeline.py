from .machine_learning import linear_svc, logistic_regression, decision_tree, random_forest, extra_trees, adaboost, xgboost, lightgbm, catboost, knn
from .deep_learning import ann


def machine_learning():
    #logistic_regression.train()
    #decision_tree.train()
    #random_forest.train()
    #extra_trees.train()
    #adaboost.train()
    #xgboost.train()
    #lightgbm.train()
    #catboost.train()
    linear_svc.train()
    # knn.train()


def deep_learning():
    ann.train()


def main():
    deep_learning()
    

if __name__ == "__main__":
    main()