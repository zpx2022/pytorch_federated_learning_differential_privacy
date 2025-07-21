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
        """
        Aggregates models and updates the server control variate.
        Model aggregation is handled by the parent class's agg method.
        """
        if not self.selected_clients or self.n_data == 0:
            return self.model.state_dict(), 0

        # --- Part 1: Perform model aggregation by calling the parent's method ---
        # The super().agg() call executes FedAvg and updates self.model.
        # It also increments self.round, so we don't need to do it again here.
        model_state, avg_loss, _ = super().agg()

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

        # self.round is already incremented in super().agg()
        return self.model.state_dict(), avg_loss

    def rec(self, name, state_dict, n_data, loss, ccv_delta):
        """SCAFFOLD-specific receive method for the control variate delta."""
        super().rec(name, state_dict, n_data, loss)
        self.client_ccv_deltas[name] = ccv_delta

    def flush(self):
        """Clears SCAFFOLD-specific client info for the next round."""
        super().flush()
        self.client_ccv_deltas = {}
