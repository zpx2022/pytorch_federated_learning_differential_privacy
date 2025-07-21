# File: fed_baselines/client_base.py

import torch
import torch.nn as nn
import torch.nn.utils as utils
import numpy as np
import copy
from torch.utils.data import DataLoader
from utils.fed_utils import assign_dataset, init_model

class FedClient(object):
    """
    The base class for a federated learning client.
    The `train` method is implemented as a Template Method, defining the standard local training workflow.
    Subclasses can override the hook methods to customize algorithm-specific behaviors.
    """
    def __init__(self, name, dataset_id, model_name, config):
        # Original __init__ code remains unchanged
        self.target_ip = '127.0.0.3'
        self.port = 9999
        self.name = name
        self._epoch = config['num_local_epoch']
        self._batch_size = config['batch_size']
        self._lr = config['lr']
        self._momentum = config['momentum']
        self._use_ldp = config['use_ldp']
        self._grad_clip_norm = config['grad_clip_norm']
        if self._use_ldp:
            self._laplace_noise_scale = config['laplace_noise_scale']
        self.n_data = 0
        self.trainset = None
        self.test_data = None
        self._num_class, self._image_dim, self._image_channel = assign_dataset(dataset_id)
        self.model_name = model_name
        self.model = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        gpu = 0
        self._device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu != -1 else "cpu")

    def load_trainset(self, trainset):
        """Client loads its local training dataset."""
        self.trainset = trainset
        self.n_data = len(trainset)

    def update(self, model_state_dict, **kwargs):
        """Client updates its model from the server's global model."""
        self.model.load_state_dict(model_state_dict)

    def _apply_ldp(self):
        """
        Applies the Local Differential Privacy mechanism to the model's gradients.
        This includes gradient clipping and adding Laplacian noise.
        """
        utils.clip_grad_norm_(self.model.parameters(), max_norm=self._grad_clip_norm)
        if self._use_ldp:      
            laplace_dist = torch.distributions.laplace.Laplace(0, self._laplace_noise_scale)       
            for param in self.model.parameters():
                if param.grad is not None:
                    noise = laplace_dist.sample(param.grad.size()).to(self._device)
                    param.grad.data.add_(noise)

    # --- Hook Methods for subclasses to override ---
    def _before_train(self):
        """Hook for preparations before the training loop starts."""
        self.global_model_state_before_train = copy.deepcopy(self.model.state_dict())
        self.num_steps = 0
        self.latest_loss = 0.0

    def _calculate_custom_loss(self, standard_loss):
        """
        Hook to calculate or modify the loss.
        FedProx will override this to add its proximal term.
        """
        return standard_loss

    def _perform_gradient_correction(self):
        """
        Hook to perform gradient correction after loss.backward().
        SCAFFOLD will override this.
        """
        pass

    def _prepare_results(self):
        """
        Hook to package and return the final results after training.
        Each algorithm might have a different return signature.
        """
        return self.model.state_dict(), self.n_data, self.latest_loss

    # --- The Template Method ---
    def train(self):
        """
        Client trains the model on its local dataset using a generic workflow.
        This method defines a fixed training skeleton and calls hook methods
        to execute algorithm-specific logic.
        """
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True)

        self.model.to(self._device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        loss_func = nn.CrossEntropyLoss()

        # Call the pre-training preparation hook
        self._before_train()

        for epoch in range(self._epoch):
            for step, (x, y) in enumerate(train_loader):
                b_x, b_y = x.to(self._device), y.to(self._device)

                with torch.enable_grad():
                    self.model.train()
                    output = self.model(b_x)
                    
                    # 1. Calculate standard loss
                    loss = loss_func(output, b_y.long())
                    
                    # 2. (Hook) Calculate custom loss
                    loss = self._calculate_custom_loss(loss)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # 3. (Hook) Perform gradient correction
                    self._perform_gradient_correction()

                    # 4. Apply Local Differential Privacy
                    self._apply_ldp()
                    
                    optimizer.step()
                    self.latest_loss = loss.item()
                    self.num_steps += 1
        
        # 5. (Hook) Prepare and return final results
        return self._prepare_results()
