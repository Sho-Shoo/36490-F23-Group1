import pickle
from tqdm import tqdm
from classes.data_loader import DataLoader
from classes.nn5 import NN5
from classes.utils import train_nn, evaluate_nn
from classes.pytorch_data_loader import USAPytorchDataloader
import torch
from torch.nn.modules.loss import _Loss as Loss
from classes.utils import r2oos

if __name__ == "__main__":

    data = DataLoader("data/usa_short.csv")
    print(f"Data is loaded!")

    YEAR = 10000
    validation_r2s = []
    test_r2s = []
    predictions = []
    model_states = []

    for train_start in tqdm(range(19800101, 20000101 + 2 * YEAR, YEAR)):
        train_end = train_start + 10 * YEAR
        validate_start = train_end
        validate_end = validate_start + 5 * YEAR
        test_start = validate_end
        test_end = test_start + YEAR

        nn5 = NN5()
        train_dataloader = USAPytorchDataloader(data, train_start, train_end, batch_size=1024)
        validation_dataloader = USAPytorchDataloader(data, validate_start, validate_end, batch_size=1024)
        loss_fn: Loss = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(params=nn5.parameters(), lr=0.001, momentum=0.9)
        # r2_metric: Metric = torchmetrics.R2Score()
        r2_metric = r2oos

        model_state, train_r2, val_r2 = train_nn(nn5, train_dataloader, validation_dataloader,
                                                 optimizer, loss_fn, r2_metric,
                                                 "100-epoch run", "nn5", EPOCHS=100)

        test_r2, prediction = evaluate_nn(NN5, model_state, data, test_start, test_end, r2_metric)

        validation_r2s.append(val_r2)
        test_r2s.extend(test_r2)
        predictions.extend(prediction)
        model_states.append(model_state)

    output_dir = "outputs/nn5_100epochs/"

    try:
        with open(output_dir + "test_r2s.pkl", "wb") as f:
            pickle.dump(test_r2s, f)
    except:
        print(f"test_r2s is {test_r2s}")

    try:
        with open(output_dir + "validation_r2s.pkl", "wb") as f:
            pickle.dump(validation_r2s, f)
    except:
        print(f"validation_r2s is {validation_r2s}")

    try:
        with open(output_dir + "predictions.pkl", "wb") as f:
            pickle.dump(predictions, f)
    except:
        print(f"predictions is {predictions}")

    try:
        with open(output_dir + "model_states.pkl", "wb") as f:
            pickle.dump(model_states, f)
    except:
        print(f"model_states is {model_states}")
