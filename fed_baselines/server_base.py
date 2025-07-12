import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from utils.models import *
from utils.fed_utils import assign_dataset, init_model

class FedServer(object):
    """
    The base class for a federated learning server.
    It handles model testing, client selection, and standard FedAvg aggregation.
    """
    def __init__(self, client_list, dataset_id, model_name, config):
        # Dictionaries to store updates from clients
        self.client_state = {}
        self.client_loss = {}
        self.client_n_data = {}
        self.selected_clients = []
        
        self._batch_size = config['batch_size']
        self.client_list = client_list
        self.testset = None
        self.round = 0
        self.n_data = 0
        self._dataset_id = dataset_id

        # Set device for testing
        gpu = 0
        self._device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu != -1 else "cpu")

        # Initialize the global model
        self._num_class, self._image_dim, self._image_channel = assign_dataset(dataset_id)
        self.model_name = model_name
        self.model = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)

    def load_testset(self, testset):
        """Server loads the global test dataset."""
        self.testset = testset

    def state_dict(self):
        """Server returns the current global model state."""
        return self.model.state_dict()

    def test(self):
        """Server tests the global model on the test dataset."""
        test_loader = DataLoader(self.testset, batch_size=self._batch_size, shuffle=False)
        self.model.to(self._device)
        self.model.eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self._device), targets.to(self._device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = correct / total
        return accuracy

    def select_clients(self, connection_ratio=1):
        """Selects all clients for simplicity in this implementation."""
        self.selected_clients = self.client_list
        self.n_data = sum(self.client_n_data.get(client_id, 0) for client_id in self.client_list)

    def agg(self):
        """Aggregates models from clients using the FedAvg algorithm."""
        client_num = len(self.selected_clients)
        self.n_data = sum(self.client_n_data.values())

        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0, 0

        # Initialize a new model state for aggregation
        model_state = self.model.state_dict()
        for key in model_state:
            model_state[key] = torch.zeros_like(model_state[key])

        avg_loss = 0.0
        for name in self.selected_clients:
            if name in self.client_state:
                weight = self.client_n_data[name] / self.n_data
                for key in self.client_state[name]:
                    model_state[key] += self.client_state[name][key] * weight
                avg_loss += self.client_loss[name] * weight
        
        self.model.load_state_dict(model_state)
        self.round += 1
        return model_state, avg_loss, self.n_data

    def rec(self, name, state_dict, n_data, loss):
        """Server receives local updates from a client."""
        self.client_state[name] = state_dict
        self.client_n_data[name] = n_data
        self.client_loss[name] = loss

    def flush(self):
        """Clears client information for the next round."""
        self.n_data = 0
        self.client_state = {}
        self.client_n_data = {}
        self.client_loss = {}
