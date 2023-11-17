import pickle
import numpy as np
from tqdm import tqdm
from classes.data_loader import DataLoader
from classes.random_forest import RandomForest


if __name__ == "__main__":

    data = DataLoader("data/usa_short.csv")
    print(f"Data is loaded!")

    N_ESTIMATORS_VALUES = list(np.logspace(100, 1000, 50)) 
    MAX_DEPTH_VALUES = list(np.linspace(50, 500, 25))  # [0.0, 0.1, ..., 1.0]

    YEAR = 10000

    validation_r2s = []
    test_r2s = []
    predictions = []
    n_estimators = []
    max_depths = []
    models = []

    for train_start in tqdm(range(19800101, 20000101 + 2 * YEAR, YEAR)):
        train_end = train_start + 10 * YEAR
        validate_start = train_end
        validate_end = validate_start + 5 * YEAR
        test_start = validate_end
        test_end = test_start + YEAR

        best_model, best_r2, best_n_estimator, best_max_depth = RandomForest.validate(data, train_start, train_end, 
                                                    validate_start, validate_end, N_ESTIMATORS_VALUES, MAX_DEPTH_VALUES)
        validation_r2s.append(best_r2)
        n_estimators.append(best_n_estimator)
        max_depths.append(best_max_depth)
        models.append(best_model)

        test_r2, prediction = RandomForest.evaluate(data, best_model, test_start, test_end)
        test_r2s.extend(test_r2)
        predictions.extend(prediction)

    try:
        with open("outputs/rf/test_r2s.pkl", "wb") as f:
            pickle.dump(test_r2s, f)
    except:
        print(f"test_r2s is {test_r2s}")

    try:
        with open("outputs/rf/validation_r2s.pkl", "wb") as f:
            pickle.dump(validation_r2s, f)
    except:
        print(f"validation_r2s is {validation_r2s}")

    try:
        with open("outputs/rf/predictions.pkl", "wb") as f:
            pickle.dump(predictions, f)
    except:
        print(f"predictions is {predictions}")
    try:
        with open('outputs/rf/models.pkl', 'wb') as f:
            pickle.dump(models, f)
    except:
        print(f"models: {models}")
    try:
        with open('outputs/rf/n_estimators.pkl', 'wb') as f:
            pickle.dump(n_estimators, f)
    except:
        print(f"alphas: {n_estimators}")