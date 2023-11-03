import pickle
from tqdm import tqdm
from classes.data_loader import DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

if __name__ == "__main__":

    # data = DataLoader("data/usa_short.csv")
    data = DataLoader("usa1.csv")
    print(f"Data is loaded!")

    YEAR = 10000
    validation_r2s = []
    test_r2s = []
    predictions = []

    for train_start in tqdm(range(19800101, 20000101 + 2 * YEAR, YEAR)):
        train_end = train_start + 10 * YEAR
        validate_start = train_end
        validate_end = validate_start + 5 * YEAR
        test_start = validate_end
        test_end = test_start + YEAR

        train_data = data.slice(train_start, train_end)
        val_data = data.slice(validate_start, validate_end)
        test_data = data.slice(test_start, test_end)

        X_train, y_train = DataLoader.get_x(train_data), DataLoader.get_y(train_data)
        X_val, y_val = DataLoader.get_x(val_data), DataLoader.get_y(val_data)
        X_test, y_test = DataLoader.get_x(test_data), DataLoader.get_y(test_data)

        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(X_train, y_train)

        val_r2 = r2_score(y_val, rf.predict(X_val))
        test_r2 = r2_score(y_test, rf.predict(X_test))

        validation_r2s.append(val_r2)
        test_r2s.append(test_r2)
        predictions.extend(rf.predict(X_test))

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