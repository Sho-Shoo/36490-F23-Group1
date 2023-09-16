import sys
from . import load_data
from . import lasso_model


def main():
    if len(sys.argv) != 5:
        print("Wrong number of arguments. Arguments: file path, start date, end date, lambda (Date format: YYYYMMDD)")

    file_path = sys.argv[1]
    start_date = int(sys.argv[2])
    end_date = int(sys.argv[3])
    lambda_val = int(sys.arg[4])
    print("Running LASSO model for \n")
    print("***** File: {file_path}")
    print("***** Lambda: {sys.arg[4]}")

    data = data_loader(file_path)

    # split date to train 10 years, validate 5 years, test 1 year
    # each: (start, end) end exclusive
    for year in range(start_date, end_date - 15):
        train_start, train_end = year, year + 10
        validation_start, validation_end = year + 10, year + 15
        test_start, test_end = year + 15, year + 16

        model = lasso_model(data, lambda_val)
        print(
            f"Training lasso model from {str(start_date)} to {str(end_date)}...")
        model.fit(train_start, train_end)

        best_param = model.evaluate(validation_start, validation_end)
        print(
            f"Result for validation from {str(validation_start)} to {str(validation_end)}: {str(best_param)}")
        # ? do we need other types of test results?
        test_performance = model.evaluate(test_start, test_end)
        print(
            f"Result for test from {str(test_start)} to {str(test_end)}: {str(test_performance)}")


if __name__ == "__main__":
    main()
