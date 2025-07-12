import torch
import torch.nn as nn
import torch.nn.utils as utils
import copy
from torch.utils.data import DataLoader
from fed_baselines.client_base import FedClient 

class FedProxClient(FedClient):
    """Client implementation for the FedProx algorithm."""
    
    def __init__(self, name, dataset_id, model_name, config):
        super().__init__(name, dataset_id, model_name, config)
        # Get the mu parameter for FedProx from config, with a default value
        self.mu = config.get('mu', 0.1)

    def train(self):
        """Performs local training using FedProx logic."""
        if self.trainset is None:
            raise ValueError("Training dataset not loaded.")
        
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True)

        self.model.to(self._device)
        # Keep a copy of the global model weights for the proximal term
        global_weights = copy.deepcopy(list(self.model.parameters()))

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        loss_func = nn.CrossEntropyLoss()
        
        latest_loss = 0.0
        for epoch in range(self._epoch):
            for step, (x, y) in enumerate(train_loader):
                b_x, b_y = x.to(self._device), y.to(self._device)

                with torch.enable_grad():
                    self.model.train()
                    output = self.model(b_x)
                    
                    loss = loss_func(output, b_y.long())
                    
                    # --- FedProx core logic ---
                    # Add the proximal term to the standard loss
                    prox_term = 0.0
                    for param, global_param in zip(self.model.parameters(), global_weights):
                        prox_term += (self.mu / 2) * torch.norm((param - global_param)) ** 2
                    loss += prox_term

                    optimizer.zero_grad()
                    loss.backward()

                    # --- LDP block ---
                    utils.clip_grad_norm_(self.model.parameters(), max_norm=self._grad_clip_norm)
                    if self._use_ldp:
                        laplace_dist = torch.distributions.laplace.Laplace(0, self._laplace_noise_scale)
                        for param in self.model.parameters():
                            if param.grad is not None:
                                noise = laplace_dist.sample(param.grad.size()).to(self._device)
                                param.grad.data.add_(noise)

                    optimizer.step()
                    latest_loss = loss.item()

        return self.model.state_dict(), self.n_data, latest_loss
