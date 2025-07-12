import torch
import torch.nn as nn
import torch.nn.utils as utils
import copy
from torch.utils.data import DataLoader
from fed_baselines.client_base import FedClient

class FedNovaClient(FedClient):
    """Client implementation for the FedNova algorithm."""

    def __init__(self, name, dataset_id, model_name, config):
        super().__init__(name, dataset_id, model_name, config)
        # Get the rho parameter for FedNova from config, with a default value
        self.rho = config.get('rho', 0.9)
        self._momentum = self.rho

    def train(self):
        """Performs local training using FedNova logic."""
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True)

        self.model.to(self._device)
        global_weights = copy.deepcopy(self.model.state_dict())

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        loss_func = nn.CrossEntropyLoss()

        tau = 0
        latest_loss = 0.0
        for epoch in range(self._epoch):
            for step, (x, y) in enumerate(train_loader):
                b_x, b_y = x.to(self._device), y.to(self._device)

                with torch.enable_grad():
                    self.model.train()
                    output = self.model(b_x)
                    loss = loss_func(output, b_y.long())
                    
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
                    tau += 1
                    latest_loss = loss.item()

        # --- FedNova core logic ---
        # Calculate the normalization coefficient
        coeff = (tau - self.rho * (1 - pow(self.rho, tau)) / (1 - self.rho)) / (1 - self.rho) if self.rho != 1.0 else tau
        
        # Calculate the normalized gradients
        state_dict = self.model.state_dict()
        norm_grad = copy.deepcopy(global_weights)
        for key in norm_grad:
            if coeff != 0:
                norm_grad[key] = torch.div(global_weights[key] - state_dict[key], coeff)
            else:
                norm_grad[key] = torch.zeros_like(norm_grad[key])

        return self.model.state_dict(), self.n_data, latest_loss, coeff, norm_grad
