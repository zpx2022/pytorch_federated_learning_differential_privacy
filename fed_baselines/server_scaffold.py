import copy
import torch
from fed_baselines.server_base import FedServer
from utils.fed_utils import init_model

class ScaffoldServer(FedServer):
    """Server implementation for the SCAFFOLD algorithm."""

    def __init__(self, client_list, dataset_id, model_name, config):
        super().__init__(client_list, dataset_id, model_name, config)
        # Initialize the server control variate (c)
        self.scv = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        for param in self.scv.parameters():
            param.data.zero_()
        # Dictionary to store client control variate deltas
        self.client_ccv_deltas = {}

    def agg(self):
        """Aggregates models and updates the server control variate."""
        if not self.selected_clients or self.n_data == 0:
            return self.model.state_dict(), 0, {}

        # --- Part 1: Aggregate client models (standard FedAvg) ---
        avg_loss = 0.0
        model_state = self.model.state_dict()
        for key in model_state:
            model_state[key] = torch.zeros_like(model_state[key])

        for name in self.selected_clients:
            if name in self.client_state:
                weight = self.client_n_data[name] / self.n_data
                for key in model_state:
                    model_state[key] += self.client_state[name][key] * weight
                avg_loss += self.client_loss[name] * weight
        self.model.load_state_dict(model_state)

        # --- Part 2: Aggregate client control variate deltas (delta_c_i) ---
        total_delta_c = {key: 0.0 for key in self.scv.state_dict().keys()}
        for name in self.selected_clients:
            if name in self.client_ccv_deltas:
                for key in total_delta_c:
                    total_delta_c[key] += self.client_ccv_deltas[name][key]

        # --- Part 3: Update server control variate (c) ---
        scv_state = self.scv.state_dict()
        with torch.no_grad():
            for key in scv_state:
                 if len(self.selected_clients) > 0:
                    scv_state[key] += total_delta_c[key] / len(self.selected_clients)
        self.scv.load_state_dict(scv_state)

        self.round += 1
        return self.model.state_dict(), avg_loss, self.scv.state_dict()

    def rec(self, name, state_dict, n_data, loss, ccv_delta):
        """SCAFFOLD-specific receive method for the control variate delta."""
        super().rec(name, state_dict, n_data, loss)
        self.client_ccv_deltas[name] = ccv_delta

    def flush(self):
        """Clears SCAFFOLD-specific client info for the next round."""
        super().flush()
        self.client_ccv_deltas = {}
