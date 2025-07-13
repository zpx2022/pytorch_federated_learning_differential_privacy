import torch
import copy
from fed_baselines.client_base import FedClient

class FedNovaClient(FedClient):
    """Client implementation for the FedNova algorithm."""

    def __init__(self, name, dataset_id, model_name, config):
        super().__init__(name, dataset_id, model_name, config)
        self.rho = config.get('rho', 0.9)
        # FedNova uses rho as momentum
        self._momentum = self.rho

    def _prepare_results(self):
        """Override: After training, calculate normalized gradients and package FedNova-specific results."""
        # Calculate the normalization coefficient
        tau = self.num_steps
        coeff = (tau - self.rho * (1 - pow(self.rho, tau)) / (1 - self.rho)) / (1 - self.rho) if self.rho != 1.0 else tau
        
        # Calculate the normalized gradients
        state_dict = self.model.state_dict()
        norm_grad = copy.deepcopy(self.global_model_state_before_train)
        for key in norm_grad:
            if coeff != 0:
                norm_grad[key] = torch.div(self.global_model_state_before_train[key] - state_dict[key], coeff)
            else:
                norm_grad[key] = torch.zeros_like(norm_grad[key])

        return self.model.state_dict(), self.n_data, self.latest_loss, coeff, norm_grad
