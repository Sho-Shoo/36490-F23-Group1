from classes.data_loader import DataLoader
from classes.elasticNet_model import ElasticNet_Model
from tqdm import tqdm
import pickle
import numpy as np

data = DataLoader("data/usa_short.csv")
print(f"Data is loaded!")

ALPHA_VALUES = np.logspace(-4, 4, 9) # [0.0001, 0.001, ..., 10000]
L1_RATIO_VALUES = np.linspace(0, 1, 11)  # [0.0, 0.1, ..., 1.0]
YEAR = 10000
validation_r2s = []
test_r2s = []
alphas = []
l1_ratios = []
models = []

for train_start in tqdm(range(19800101, 20000101 + 2 * YEAR, YEAR)):
    train_end = train_start + 10 * YEAR
    validate_start = train_end
    validate_end = validate_start + 5 * YEAR
    test_start = validate_end
    test_end = test_start + YEAR

    # perform grid search for the best alpha and l1_ratio hyperparameters
    best_model, best_r2, best_alpha, best_l1_ratio = ElasticNet_Model.validate(data, train_start, validate_end, ALPHA_VALUES, L1_RATIO_VALUES)
    validation_r2s.append(best_r2)
    alphas.append(best_alpha)
    l1_ratios.append(best_l1_ratio)
    models.append(best_model)

    # testing on an extra year of data
    test_r2 = ElasticNet_Model.evaluate(data, best_model, test_start, test_end)
    test_r2s.extend(test_r2)

try:
    with open('outputs/elasticnet/test_r2s.pkl', 'wb') as f:
        pickle.dump(test_r2s, f)
except:
    print(f"test_r2s: {test_r2s}")

try:
    with open('outputs/elasticnet/validation_r2s.pkl', 'wb') as f:
        pickle.dump(validation_r2s, f)
except:
    print(f"validation_r2s: {validation_r2s}")

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

print("Elastic Net training script successfully finished!")
