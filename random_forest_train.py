import pickle
from tqdm import tqdm
from classes.data_loader import DataLoader
from classes.random_forest import RandomForest


if __name__ == "__main__":

    data = DataLoader("data/usa_short.csv")
    print(f"Data is loaded!")

    ALPHA_VALUES = list(np.logspace(-4, 4, 9)) # [0.0001, 0.001, ..., 10000]
    L1_RATIO_VALUES = list(np.linspace(0, 1, 11))  # [0.0, 0.1, ..., 1.0]
    YEAR = 10000

    validation_r2s = []
    test_r2s = []
    predictions = []

    alphas = []
    l1_ratios = []
    models = []

    for train_start in tqdm(range(19800101, 20000101 + 2 * YEAR, YEAR)):
        train_end = train_start + 10 * YEAR
        validate_start = train_end
        validate_end = validate_start + 5 * YEAR
        test_start = validate_end
        test_end = test_start + YEAR

        best_model, best_r2, best_alpha, best_l1_ratio = RandomForest.validate(data, train_start, train_end,
                                                                               validate_start, validate_end,
                                                                               ALPHA_VALUES, L1_RATIO_VALUES)
        validation_r2s.append(best_r2)
        alphas.append(best_alpha)
        l1_ratios.append(best_l1_ratio)
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
        with open('outputs/elasticnet/alphas.pkl', 'wb') as f:
            pickle.dump(alphas, f)
    except:
        print(f"alphas: {alphas}")

    try:
        with open('outputs/elasticnet/l1_ratios.pkl', 'wb') as f:
            pickle.dump(l1_ratios, f)
    except:
        print(f"l1_ratios: {l1_ratios}")
    try:
        with open('outputs/elasticnet/models.pkl', 'wb') as f:
            pickle.dump(models, f)
    except:
        print(f"models: {models}")