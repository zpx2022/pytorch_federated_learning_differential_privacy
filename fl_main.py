import os
import random
import json
import pickle
import argparse
import yaml
import torch
import numpy as np
from json import JSONEncoder
from tqdm import tqdm
import concurrent.futures
import multiprocessing as mp

# Import all necessary client and server classes
from fed_baselines.client_base import FedClient
from fed_baselines.client_fedprox import FedProxClient
from fed_baselines.client_scaffold import ScaffoldClient
from fed_baselines.client_fednova import FedNovaClient
from fed_baselines.server_base import FedServer
from fed_baselines.server_scaffold import ScaffoldServer
from fed_baselines.server_fednova import FedNovaServer

# Assume these local modules exist in the project structure
from postprocessing.recorder import PythonObjectEncoder, Recorder
from preprocessing.baselines_dataloader import divide_data
from utils.models import *

# --- Helper function for parallel client training ---
def client_training_process(client, global_state_dict, fed_algo, scv_state=None):
    """
    A wrapper function for a single client's training process to be executed in parallel.
    It handles algorithm-specific update calls and returns all results from the client.
    """
    # Handle the special update call for SCAFFOLD
    if fed_algo == 'SCAFFOLD' and scv_state is not None:
        client.update(global_state_dict, scv_state)
    else:
        client.update(global_state_dict)
    
    # Execute local training and get all results
    train_results = client.train()
    
    # Return client name and the complete results tuple
    return client.name, train_results

# --- Checkpoint helper functions ---
def save_checkpoint(checkpoint_path, global_round, global_state_dict, max_acc, patience_counter, recorder_res):
    """Saves a training checkpoint."""
    checkpoint = {
        'global_round': global_round,
        'global_state_dict': global_state_dict,
        'max_acc': max_acc,
        'patience_counter': patience_counter,
        'recorder_res': recorder_res
    }
    torch.save(checkpoint, checkpoint_path)
    # print(f"Checkpoint saved to {checkpoint_path} at round {global_round}") # ⭐️ MODIFICATION: Commented out to keep the progress bar clean

def load_checkpoint(checkpoint_path, fed_server, recorder):
    """Loads a training checkpoint if it exists."""
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        
        global_round = checkpoint['global_round']
        global_state_dict = checkpoint['global_state_dict']
        max_acc = checkpoint['max_acc']
        patience_counter = checkpoint['patience_counter']
        recorder.res = checkpoint['recorder_res']

        fed_server.model.load_state_dict(global_state_dict)
        print(f"Checkpoint loaded. Resuming from round {global_round + 1}.")
        return global_round, global_state_dict, max_acc, patience_counter
    else:
        print("No checkpoint found. Starting training from scratch.")
        return 0, fed_server.state_dict(), 0, 0

# --- Main Program ---
def fed_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Yaml file for configuration')
    args = parser.parse_args()
    return args
    
def fed_run():
    """Main function to run the federated learning framework."""
    args = fed_args()
    with open(args.config, "r") as yaml_file:
        try:
            config = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)
    
    # --- 1. Configuration and Initialization ---
    system_config = config["system"]
    client_config = config["client"]
    fed_algo = client_config["fed_algo"]
    
    # Set random seeds for reproducibility
    np.random.seed(system_config["i_seed"])
    torch.manual_seed(system_config["i_seed"])
    random.seed(system_config["i_seed"])

    client_dict = {}
    recorder = Recorder()
    trainset_config, testset = divide_data(
        num_client=system_config["num_client"], 
        num_local_class=system_config["num_local_class"], 
        dataset_name=system_config["dataset"],
        i_seed=system_config["i_seed"]
    )
    
    # Initialize clients based on the selected algorithm
    for client_id in trainset_config['users']:
        if fed_algo == 'FedAvg':
            client_dict[client_id] = FedClient(client_id, dataset_id=system_config["dataset"], model_name=system_config["model"], config=client_config)
        elif fed_algo == 'FedProx':
            client_dict[client_id] = FedProxClient(client_id, dataset_id=system_config["dataset"], model_name=system_config["model"], config=client_config)
        elif fed_algo == 'SCAFFOLD':
            client_dict[client_id] = ScaffoldClient(client_id, dataset_id=system_config["dataset"], model_name=system_config["model"], config=client_config)
        elif fed_algo == 'FedNova':
            client_dict[client_id] = FedNovaClient(client_id, dataset_id=system_config["dataset"], model_name=system_config["model"], config=client_config)
        client_dict[client_id].load_trainset(trainset_config['user_data'][client_id])

    # Initialize the server based on the selected algorithm
    if fed_algo in ['FedAvg', 'FedProx']:
        fed_server = FedServer(trainset_config['users'], dataset_id=system_config["dataset"], model_name=system_config["model"], config=client_config)
    elif fed_algo == 'SCAFFOLD':
        fed_server = ScaffoldServer(trainset_config['users'], dataset_id=system_config["dataset"], model_name=system_config["model"], config=client_config)
    elif fed_algo == 'FedNova':
        fed_server = FedNovaServer(trainset_config['users'], dataset_id=system_config["dataset"], model_name=system_config["model"], config=client_config)
    
    fed_server.load_testset(testset)
    early_stopping_patience = client_config.get('early_stopping_patience', 100)

    # --- 2. Prepare Filenames and Paths ---
    if not os.path.exists(system_config["res_root"]):
        os.makedirs(system_config["res_root"])
    if not os.path.exists(system_config["check_root"]):
        os.makedirs(system_config["check_root"])

    base_file_name = (f"['{client_config['fed_algo']}',"
                      f"'{system_config['model']}',"
                      f"{client_config['num_local_epoch']},"
                      f"{system_config['num_local_class']},"
                      f"{client_config['use_ldp']},"
                      f"{client_config['laplace_noise_scale']}]")

    result_file_path = os.path.join(system_config["res_root"], base_file_name + "_results.json")
    checkpoint_file_path = os.path.join(system_config["check_root"], base_file_name + "_checkpoint.pth")

    # --- 3. Load Checkpoint ---
    initial_global_round, global_state_dict, max_acc, patience_counter = load_checkpoint(checkpoint_file_path, fed_server, recorder)
    global_round_start = initial_global_round + 1 if initial_global_round > 0 else 0

    # --- 4. Federated Learning Main Loop ---
    pbar = tqdm(range(global_round_start, system_config["num_round"]), initial=global_round_start, total=system_config["num_round"], desc="Federated Learning Training")
    for global_round in pbar:
        
        # Prepare extra arguments if needed (for SCAFFOLD)
        scv_state = fed_server.scv.state_dict() if fed_algo == 'SCAFFOLD' else None

        # Execute client training in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count()//2) as executor:
            futures = [executor.submit(client_training_process, 
                                       client_dict[client_id], 
                                       global_state_dict,
                                       fed_algo,
                                       scv_state)
                       for client_id in trainset_config['users']]
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    c_name, train_results = future.result()
                    
                    # Unpack results and pass them to the server based on the algorithm
                    if fed_algo in ['FedAvg', 'FedProx']:
                        s_dict, n_data, loss = train_results
                        fed_server.rec(c_name, s_dict, n_data, loss)
                    elif fed_algo == 'SCAFFOLD':
                        s_dict, n_data, loss, ccv_delta = train_results
                        fed_server.rec(c_name, s_dict, n_data, loss, ccv_delta)
                    elif fed_algo == 'FedNova':
                        s_dict, n_data, loss, coeff, norm_grad = train_results
                        fed_server.rec(c_name, s_dict, n_data, loss, coeff, norm_grad)
                except Exception as e:
                    print(f"A client training process failed: {e}")

        # Perform global aggregation on the server
        fed_server.select_clients()
        global_state_dict, avg_loss, _ = fed_server.agg()
        
        # Test the new global model
        accuracy = fed_server.test()
        fed_server.flush()

        # Record results
        recorder.res['server']['iid_accuracy'].append(float(accuracy))
        recorder.res['server']['train_loss'].append(float(avg_loss))

        # Early stopping logic
        if accuracy > max_acc:
            max_acc = accuracy
            patience_counter = 0
        else:
            patience_counter += 1

        # Update progress bar description
        pbar.set_description(
            f'Round: {global_round + 1}| Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}| Max Acc: {max_acc:.4f}| Patience: {patience_counter}'
        )
        
        # --- 5. Save Results and Checkpoint ---
        with open(result_file_path, "w") as jsfile:
            json.dump(recorder.res, jsfile, cls=PythonObjectEncoder)
        
        save_checkpoint(checkpoint_file_path, global_round, global_state_dict, max_acc, patience_counter, recorder.res)
        
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping at round {global_round+1}.")
            break 

    print(f"\nTraining finished. Results saved to {result_file_path}")

if __name__ == "__main__":
    fed_run()
