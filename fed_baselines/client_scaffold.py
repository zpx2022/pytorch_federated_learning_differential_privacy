import torch
import torch.nn as nn
import torch.nn.utils as utils
import copy
from torch.utils.data import DataLoader
from fed_baselines.client_base import FedClient
from utils.fed_utils import init_model

class ScaffoldClient(FedClient):
    """Client implementation for the SCAFFOLD algorithm."""

    def __init__(self, name, dataset_id, model_name, config):
        super().__init__(name, dataset_id, model_name, config)
        # Server control variate (c)
        self.scv = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        # Client control variate (c_i)
        self.ccv = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        # Initialize client control variate to zeros
        for param in self.ccv.parameters():
            param.data.zero_()

    def update(self, model_state_dict, scv_state):
        """SCAFFOLD-specific update method that also receives the server control variate."""
        self.model.load_state_dict(model_state_dict)
        self.scv.load_state_dict(scv_state)

    def train(self):
        """Performs local training using SCAFFOLD logic."""
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True)

        self.model.to(self._device)
        self.ccv.to(self._device)
        self.scv.to(self._device)
        
        global_model_state_before_train = copy.deepcopy(self.model.state_dict())
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        loss_func = nn.CrossEntropyLoss()
        
        latest_loss = 0.0
        num_steps = 0
        for epoch in range(self._epoch):
            for step, (x, y) in enumerate(train_loader):
                b_x, b_y = x.to(self._device), y.to(self._device)

                with torch.enable_grad():
                    self.model.train()
                    output = self.model(b_x)
                    loss = loss_func(output, b_y.long())
                    
                    optimizer.zero_grad()
                    loss.backward()

                    # --- SCAFFOLD core logic: correct gradients with control variates ---
                    for param, scv_param, ccv_param in zip(self.model.parameters(), self.scv.parameters(), self.ccv.parameters()):
                        if param.grad is not None:
                            param.grad.data += scv_param.data.to(self._device) - ccv_param.data.to(self._device)

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
                    num_steps += 1

        # --- SCAFFOLD: Update client control variate and calculate its delta ---
        new_ccv_state = self.ccv.state_dict()
        final_model_state = self.model.state_dict()
        with torch.no_grad():
            for key in new_ccv_state:
                term1 = global_model_state_before_train[key] - final_model_state[key]
                new_ccv_state[key] = self.ccv.state_dict()[key] - self.scv.state_dict()[key] + (term1 / (num_steps * self._lr))
        
        # The delta is sent to the server for aggregation
        delta_c_i = {key: new_ccv_state[key] - self.ccv.state_dict()[key] for key in new_ccv_state}
        self.ccv.load_state_dict(new_ccv_state)

        return self.model.state_dict(), self.n_data, latest_loss, delta_c_i
