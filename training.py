import torch
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np
import optuna
import csv
import os

from torch.utils.data import DataLoader, random_split
from algorithm import AdaptiveMultiStageStressTestedHPO
from model import BassTranscriptionCNN
from dataset_loader import Dataset

try:
    from google.colab import drive
except ImportError:
    print("You are not in a Google Colab environment. Saving locally")

def evaluate_model(config, train_loader, val_loader, stress_test, profile_latency):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model(config).to(device)

    criterion_class = nn.CrossEntropyLoss()
    criterion_binary = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    max_epochs = 15
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0

    if train_loader is not None and val_loader is not None:
        for epoch in range(max_epochs):
            model.train()
            for batch_audio, labels in train_loader:
                batch_audio= batch_audio.to(device)
                out_string, out_fret, out_pitch, out_onset, out_offset = model(batch_audio)

                loss_string = criterion_class(out_string, labels['string'])
                loss_fret = criterion_class(out_fret, labels['fret'])
                loss_pitch = criterion_class(out_pitch, labels['pitch'])
                loss_onset = criterion_binary(out_onset, labels['onset'])
                loss_offset = criterion_binary(out_offset, labels['offset'])

                total_loss = loss_string + loss_fret + loss_pitch + loss_onset + loss_offset

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            model.eval()
            val_loss_accumulated = 0.0
            with torch.no_grad():
                for val_audio, val_labels in val_loader:
                    val_audio = val_audio.to(device)
                    val_labels = {k: v.to(device) for k, v in val_labels.items()}

                    val_out_string, val_out_fret, val_out_pitch, val_out_onset, val_out_offset =  model(val_audio)

                    val_loss_string = criterion_class(val_out_string, val_labels['string'])
                    val_loss_fret = criterion_class(val_out_fret, val_labels['fret'])
                    val_loss_pitch = criterion_class(val_out_pitch, val_labels['pitch'])
                    val_loss_onset = criterion_binary(val_out_onset, val_labels['onset'])
                    val_loss_offset = criterion_binary(val_out_offset, val_labels['offset'])\

                    val_loss_accumulated += (val_loss_string + val_loss_fret + val_loss_pitch + val_loss_onset + val_loss_offset).item()
            
            epoch_val_loss = val_loss_accumulated / len(val_loader)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early Stopping triggerd at Epoch {epoch + 1}. No improvement since last {patience} epochs.")
                break

    validation_loss = best_val_loss if best_val_loss != float('inf') else 1.0
    avg_latency = 0.0

    if profile_latency:
        model.eval()
        dummy_input = torch.randn(1, 4608).to(device)

        latencies = []

        with torch.no_grad():
            for _ in range(10):
                if stress_test:
                    noise_std = 0.05
                    network_input = dummy_input +  torch.randn_like(dummy_input) * noise_std
                else:
                    network_input = dummy_input

                t_input = time.perf_counter()
                _ = model(network_input)
                t_output = time.perf_counter()

                latencies.append((t_output - t_input) * 1000)
        avg_latency = sum(latencies) / len(latencies)

    return validation_loss, avg_latency

def create_optuna_objective(train_loader, val_loader):
    def optuna_objective(trial):
        config = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'activation_funciton': trial.suggest_categorical('activation', ['ReLU', 'Tanh', 'ELU']),
            'convolution_layers': trial.suggest_int('convolution_layers', 1,4),
            'filter_layers': trial.suggest_categorical('filter_layers', [16, 32, 64, 128]),
            'filter_size': trial.suggest_categorical('filter_size', [2, 3, 5, 7])
        }

        loss, latency = evaluate_model(config, train_loader, val_loader, stress_test=False, profile_latency=False)
        return 1.0 - loss
    return optuna_objective

def optuna_csv_logger(study, trial, filename):
    file_exist = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)

        if not file_exist:
            writer.writerow(['Trial',
                             'Learning Rate',
                             'Dropout Rate',
                             'Activation Function',
                             'Convolution Layers',
                             'Filter Layers',
                             'Kernel Size',
                             'Fitness Score'])
        writer.writerow([trial.number + 1,
                         f"{trial.params['learning_rate']:.6f}",
                         trial.params['dropout_rate'],
                         trial.params['activation_function'],
                         trial.params['convolution_layers'],
                         trial.params['filter_layers'],
                         trial.params['filter_size'],
                         f"{trial.value:.4f}"])
        

if __name__ == "__main__":

    print("Optimization Experiment")

    try:
        drive.mount('content/drive')
        GDRIVE_PATH = "/content/drive/MyDrive/CNN Training"
    except:
        GDRIVE_PATH = "./CNN Training"
    
    if not os.path.exists(GDRIVE_PATH):
        os.makedirs(GDRIVE_PATH)
        print(f"Creating new folder at {GDRIVE_PATH}")
    else:
        print(f"Already linked in existing folder at {GDRIVE_PATH}")
    
    MODE = "PROPOSED" #   PROPOSED, BAYESIAN OPTIMIZATION, RANDOM SEARCH
    TOTAL_TRIALS = 30

    print("Loading dataset")
    full_dataset = Dataset(csv_file = "dataset_labels.csv", root_dir="./IDMT-SMT-BASS")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print(f"Training Samples: {len(train_dataset)} \nValidation Samples: {len(val_dataset)}")

    print(f"Starting optimization using {MODE} ALGORITHM")
    if MODE == "PROPOSED":
        logfile = os.path.join(GDRIVE_PATH, "hpo_proposed_log.csv")
        checkpoint = os.path.join(GDRIVE_PATH, "hpo_proposed_checkpoint.pkl")

        hpo_engine = AdaptiveMultiStageStressTestedHPO(budgetTrial=30, 
                                                       alpha=0.5, 
                                                       beta=0.5, 
                                                       rho_S=0.6, 
                                                       rho_R=0.4, 
                                                       gamma=0.3, 
                                                       logfile=logfile,
                                                       checkpoint=checkpoint)
        train_eval_wrapper = lambda config, stress_test = False: evaluate_model(config, 
                                                                                train_loader, 
                                                                                val_loader, 
                                                                                stress_test=stress_test, 
                                                                                profile_latency=True)
        best_config = hpo_engine.optimization(train_eval_wrapper)

    elif MODE in ["BAYESIAN OPTIMIZATION", "RANDOM SEARCH"]:
        database_path = f"sqlite:///{os.path.join(GDRIVE_PATH, f'optuna_{MODE.lower()}_study.db')}"
        logfile = os.path.join(GDRIVE_PATH, f"optuna_{MODE.lower()}_log.csv")

        sampler = optuna.samplers.RandomSampler() if MODE == "RANDOM SEARCH" else optuna.samplers.TPESampler()
        study = optuna.create_study(direction='maximize', 
                                    sampler=sampler,
                                    study_name=f"optimization_{MODE.lower()}",
                                    load_if_exists=True,
                                    storage=database_path)
    
        objective_function = create_optuna_objective(train_loader, val_loader)
        
        trials_left = TOTAL_TRIALS - len(study.trials)
        if trials_left > 0:
            print(f"Resuming optimization... {trials_left} trials left")
            study.optimize(objective_function, 
                        n_trials=trials_left, 
                        callbacks=[lambda s, t: optuna_csv_logger(s, t, logfile)])
        else:
            print("Optimization completed.")

        best_config = study.best_params

    print(f"Final retraining with best config : {best_config}")

    final_model = BassTranscriptionCNN(best_config).to('cuda' if torch.cuda.is_available() else 'cpu')
    model_save_path = os.path.join(GDRIVE_PATH, f"trained_{MODE.lower()}.pth")
    torch.save(final_model.state_dict(), model_save_path)

    print(f"Trained weights saved as {model_save_path}")