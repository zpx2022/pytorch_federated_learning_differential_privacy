import torch
import copy
from fed_baselines.client_base import FedClient
from utils.fed_utils import init_model

class ScaffoldClient(FedClient):
    """Client implementation for the SCAFFOLD algorithm."""

    def __init__(self, name, dataset_id, model_name, config):
        super().__init__(name, dataset_id, model_name, config)
        self.scv = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        self.ccv = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        for param in self.ccv.parameters():
            param.data.zero_()

    def update(self, model_state_dict, scv_state):
        """SCAFFOLD-specific update method that also receives the server control variate."""
        super().update(model_state_dict)
        self.scv.load_state_dict(scv_state)

    def _before_train(self):
        """Override: Also move control variates to the device."""
        super()._before_train()
        self.ccv.to(self._device)
        self.scv.to(self._device)

    def _perform_gradient_correction(self):
        """Override: Correct gradients using control variates."""
        for param, scv_param, ccv_param in zip(self.model.parameters(), self.scv.parameters(), self.ccv.parameters()):
            if param.grad is not None:
                param.grad.data += scv_param.data.to(self._device) - ccv_param.data.to(self._device)

    def _prepare_results(self):
        """Override: Update client control variate and package SCAFFOLD-specific results."""
        # Update client control variate (ccv)
        new_ccv_state = self.ccv.state_dict()
        final_model_state = self.model.state_dict()
        with torch.no_grad():
            for key in new_ccv_state:
                term1 = self.global_model_state_before_train[key] - final_model_state[key]
                new_ccv_state[key] = self.ccv.state_dict()[key] - self.scv.state_dict()[key] + (term1 / (self.num_steps * self._lr))
        
        delta_c_i = {key: new_ccv_state[key] - self.ccv.state_dict()[key] for key in new_ccv_state}
        self.ccv.load_state_dict(new_ccv_state)

        return self.model.state_dict(), self.n_data, self.latest_loss, delta_c_i
