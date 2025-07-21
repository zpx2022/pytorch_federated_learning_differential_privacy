import copy
import torch
from fed_baselines.server_base import FedServer

class FedNovaServer(FedServer):
    """Server implementation for the FedNova algorithm."""
    
    def __init__(self, client_list, dataset_id, model_name, config):
        super().__init__(client_list, dataset_id, model_name, config)
        # Dictionaries to store FedNova specific values
        self.client_coeff = {}
        self.client_norm_grad = {}

    def agg(self):
        """Aggregates models from clients using FedNova logic."""
        if not self.selected_clients or self.n_data == 0:
            return self.model.state_dict(), 0

        self.model.to(self._device)
        model_state = self.model.state_dict()

        # Aggregate normalized gradients from clients
        total_norm_grad = {key: 0.0 for key in model_state.keys()}
        total_weight = 0.0
        avg_loss = 0.0
        
        for name in self.selected_clients:
            if name in self.client_state:
                data_weight = self.client_n_data[name]
                total_weight += self.client_coeff[name] * data_weight
                for key in total_norm_grad:
                    total_norm_grad[key] += self.client_norm_grad[name][key] * data_weight
                avg_loss += self.client_loss[name] * data_weight

        # Calculate effective local steps
        effective_local_steps = total_weight / self.n_data if self.n_data != 0 else 0
        
        # Update the global model using the aggregated normalized gradients
        with torch.no_grad():
            for key in model_state:
                if self.n_data != 0:
                    model_state[key] -= effective_local_steps * (total_norm_grad[key] / self.n_data)

        self.model.load_state_dict(model_state)
        avg_loss = avg_loss / self.n_data if self.n_data != 0 else 0

        self.round += 1
        return model_state, avg_loss

    def rec(self, name, state_dict, n_data, loss, coeff, norm_grad):
        """FedNova-specific receive method for extra parameters."""
        super().rec(name, state_dict, n_data, loss)
        self.client_coeff[name] = coeff
        self.client_norm_grad[name] = norm_grad

    def flush(self):
        """Clears FedNova-specific client info for the next round."""
        super().flush()
        self.client_coeff = {}
        self.client_norm_grad = {}
