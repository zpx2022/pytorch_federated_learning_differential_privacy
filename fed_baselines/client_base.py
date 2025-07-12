import torch
import torch.nn as nn
import torch.nn.utils as utils
import numpy as np
from torch.utils.data import DataLoader
from utils.fed_utils import assign_dataset, init_model

class FedClient(object):
    """
    The base class for a federated learning client.
    It handles model initialization, data loading, and standard local training with optional LDP.
    """
    def __init__(self, name, dataset_id, model_name, config):
        # Initialize client metadata
        self.target_ip = '127.0.0.3'
        self.port = 9999
        self.name = name

        # Load parameters from the config file
        self._epoch = config['num_local_epoch']
        self._batch_size = config['batch_size']
        self._lr = config['lr']
        self._momentum = config['momentum']
        
        # Load Local Differential Privacy (LDP) parameters
        self._use_ldp = config['use_ldp']
        self._grad_clip_norm = config['grad_clip_norm']
        if self._use_ldp:
            self._laplace_noise_scale = config['laplace_noise_scale']
        
        self.n_data = 0
        self.trainset = None
        self.test_data = None

        # Initialize the local model
        self._num_class, self._image_dim, self._image_channel = assign_dataset(dataset_id)
        self.model_name = model_name
        self.model = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        
        # Set device for training
        gpu = 0
        self._device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu != -1 else "cpu")

    def load_trainset(self, trainset):
        """Client loads its local training dataset."""
        self.trainset = trainset
        self.n_data = len(trainset)

    def update(self, model_state_dict):
        """Client updates its model from the server's global model."""
        self.model.load_state_dict(model_state_dict)

    def train(self):
        """Client trains the model on its local dataset."""
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True)

        self.model.to(self._device)
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
                    
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # --- Local Differential Privacy block ---
                    # Clip gradients to a maximum norm
                    utils.clip_grad_norm_(self.model.parameters(), max_norm=self._grad_clip_norm)
                    if self._use_ldp:      
                        # Add Laplacian noise to gradients
                        laplace_dist = torch.distributions.laplace.Laplace(0, self._laplace_noise_scale)       
                        for param in self.model.parameters():
                            if param.grad is not None:
                                noise = laplace_dist.sample(param.grad.size()).to(self._device)
                                param.grad.data.add_(noise)
                    
                    optimizer.step()
                    latest_loss = loss.item()

        return self.model.state_dict(), self.n_data, latest_loss
