import pickle
from tqdm import tqdm
from classes.data_loader import DataLoader
from classes.nn3 import NN3Classifier
from classes.utils import train_nn, evaluate_nn, AccuracyMetric
from classes.pytorch_data_loader import USAPytorchDataloader
import torch
from torch.nn.modules.loss import _Loss as Loss

if __name__ == "__main__":

    data = DataLoader("data/usa_short.csv")
    print(f"Data is loaded!")

    YEAR = 10000
    validation_accs = []
    test_accs = []
    predictions = []
    model_states = []

    for train_start in tqdm(range(19800101, 20000101 + 2 * YEAR, YEAR)):
        train_end = train_start + 10 * YEAR
        validate_start = train_end
        validate_end = validate_start + 5 * YEAR
        test_start = validate_end
        test_end = test_start + YEAR

        nn3 = NN3Classifier()
        train_dataloader = USAPytorchDataloader(data, train_start, train_end, is_classifier=True, batch_size=1024)
        validation_dataloader = USAPytorchDataloader(data, validate_start, validate_end, is_classifier=True, batch_size=1024)
        loss_fn: Loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params=nn3.parameters(), lr=0.001, momentum=0.9)
        acc_metric = AccuracyMetric(task="multiclass", num_classes=3)

        model_state, train_acc, val_acc = train_nn(nn3, train_dataloader, validation_dataloader,
                                                   optimizer, loss_fn, acc_metric,
                                                   "train run", "nn3", verbose=True, EPOCHS=50)

        test_acc, prediction = evaluate_nn(NN3Classifier, model_state, data, test_start, test_end, acc_metric, is_classifier=True)

        validation_accs.append(val_acc)
        test_accs.extend(test_acc)
        predictions.extend(prediction)
        model_states.append(model_state)

    try:
        with open("outputs/nn3_classifier_50epochs/test_accs.pkl", "wb") as f:
            pickle.dump(test_accs, f)
    except:
        print(f"test_accs is {test_accs}")

    try:
        with open("outputs/nn3_classifier_50epochs/validation_accs.pkl", "wb") as f:
            pickle.dump(validation_accs, f)
    except:
        print(f"validation_accs is {validation_accs}")

    try:
        with open("outputs/nn3_classifier_50epochs/predictions.pkl", "wb") as f:
            pickle.dump(predictions, f)
    except:
        print(f"predictions is {predictions}")

    try:
        with open("outputs/nn3_classifier_50epochs/model_states.pkl", "wb") as f:
            pickle.dump(model_states, f)
    except:
        print(f"model_states is {model_states}")
