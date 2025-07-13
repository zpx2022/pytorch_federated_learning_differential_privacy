import torch
import copy
from fed_baselines.client_base import FedClient 

class FedProxClient(FedClient):
    """Client implementation for the FedProx algorithm."""
    
    def __init__(self, name, dataset_id, model_name, config):
        super().__init__(name, dataset_id, model_name, config)
        self.mu = config.get('mu', 0.1)
        self.global_params = None

    def _before_train(self):
        """Override: Save global model parameters before training."""
        super()._before_train()
        self.global_params = copy.deepcopy(list(self.model.parameters()))

    def _calculate_custom_loss(self, standard_loss):
        """Override: Add the proximal term to the standard loss."""
        prox_term = 0.0
        for param, global_param in zip(self.model.parameters(), self.global_params):
            prox_term += (self.mu / 2) * torch.norm((param - global_param)) ** 2
        return standard_loss + prox_term
