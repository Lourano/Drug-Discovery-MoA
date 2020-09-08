# /kaggle/input/kfold-for-moa/MultilabelStratifiedKFold.csv
# /kaggle/input/lish-moa/test_features.csv
# /kaggle/input/lish-moa/sample_submission.csv
# /kaggle/input/lish-moa/train_features.csv
# /kaggle/input/lish-moa/train_targets_scored.csv
# /kaggle/input/lish-moa/train_targets_nonscored.csv


from train_core.core import Dataset, Engine, FullyConnectedModel
from functools import partial
import pandas as pd
import numpy as np
import optuna
import torch

DEVICE = "cuda"
EPOCHS = 5

def add_dummies(data, column):
    ohe = pd.get_dummies(data[column])
    ohe_columns = [f"{column}_{c}" for c in ohe.columns]
    ohe.columns = ohe_columns
    data = data.drop(column, axis=1)
    data = data.join(ohe)
    return data


def process_data(data):
    data = add_dummies(data, "cp_time")
    data = add_dummies(data, "cp_dose")
    data = add_dummies(data, "cp_type")
    return data


def run_training(fold, params, save_model=False):
    df = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")
    df = process_data(df)
    folds = pd.read_csv("/kaggle/input/kfold-for-moa/MultilabelStratifiedKFold.csv")
    
    targets = folds.drop(["sig_id", "kfold"], axis=1).columns
    features = df.drop(["sig_id"], axis=1).columns
    df = df.merge(folds, on="sig_id", how="left")
    
    train_df = df[df.kfold != fold].reset_index(drop=True)
    val_df = df[df.kfold == fold].reset_index(drop=True)
    
    x_train = train_df[features].to_numpy()
    x_val = val_df[features].to_numpy()
    
    y_train = train_df[targets].to_numpy()
    y_val = val_df[targets].to_numpy()
    
    train_tensor = Dataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=128, num_workers=8)
    
    val_tensor = Dataset(x_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_tensor, batch_size=128, num_workers=8)
    
    model = FullyConnectedModel(
        num_features=x_train.shape[1],
        num_targets=y_train.shape[1],
        num_layers=params["num_layers"],
        hidden_size=params["hidden_size"],
        dropout=params["dropout"])
    
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3,
                                                           threshold=0.00001, mode='min',
                                                           verbose=True)
    engine = Engine(model, optimizer, device=DEVICE)
    
    best_loss = np.inf
    early_stopping = 10
    early_stopping_counter = 0
    
    for epoch in range(EPOCHS):
        train_loss = engine.train(train_loader)
        val_loss = engine.validate(val_loader)
        scheduler.step(val_loss)
        
        print(f"Fold = {fold}, Epoch = {epoch}, train_loss = {train_loss}, val_loss = {val_loss}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            if save_model:
                torch.save(model.state_dict(), f"model_fold{fold}.bin")
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter > early_stopping:
            break
    
    print(f"fold = {fold}, best_loss = {best_loss}")
    return best_loss


def objective(trial):
    params = {
        "num_layers": trial.suggest_int("num_layers", 4, 7),
        "hidden_size": trial.suggest_int("hidden_size", 256, 1024),
        "dropout": trial.suggest_uniform("dropout", 0.1, 0.8),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-6, 1e-3)}
    
    all_loss = []
    for fold in range(5):
        temp_loss = run_training(fold, params, save_model=False)
        all_loss.append(temp_loss)
    return np.mean(all_loss)


if __name__ == "__main__":
    
    partial_obj = partial(objective)
    study = optuna.create_study(direction="minimize")
    study.optimize(partial_obj, n_trials=150)
    
    print("Best trial:")
    trial_ = study.best_trial
    
    print("Value: {}".format(trial_.value))
    
    print("Params: ")
    best_params = trial_.params
    print(best_params)
    
    scores = 0
    for j in range(5):
        score = run_training(fold=j, params=best_params, save_model=True)
        scores += score
    
    print(f"OOF Score: {scores / 5}")


        
    
    