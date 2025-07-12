import numpy as np
import matplotlib.pyplot as plt
import json
from json import JSONEncoder
import pickle

# --- Helper classes for JSON serialization with custom Python objects ---
json_types = (list, dict, str, int, float, bool, type(None))

class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}

def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(dct['_python_object'].encode('latin-1'))
    return dct

class Recorder(object):
    """
    Handles loading and plotting of experiment results from JSON files.
    """
    def __init__(self):
        """Initializes the Recorder."""
        self.res_list = []
        self.res = {'server': {'iid_accuracy': [], 'train_loss': []},
                    'clients': {'iid_accuracy': [], 'train_loss': []}}

    def load(self, filename, label):
        """
        Loads a result file.
        :param filename: Name of the result file.
        :param label: Label for this result file to be used in plots.
        """
        with open(filename, 'r', encoding='utf-8') as json_file:
            res = json.load(json_file, object_hook=as_python_object)
        self.res_list.append((res, label))

    def plot(self):
        """
        Plots the testing accuracy and training loss curves for all loaded results.
        """
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 14))
        
        for i, (res, label) in enumerate(self.res_list):
            # Plot accuracy curve
            accuracy_curve = res['server']['iid_accuracy']
            if accuracy_curve:
                rounds = np.arange(1, len(accuracy_curve) + 1)
                line, = axes[0].plot(rounds, np.array(accuracy_curve), label=label, alpha=1, linewidth=2)

                # --- Annotation logic for max accuracy ---
                max_acc = max(accuracy_curve)
                max_acc_index = accuracy_curve.index(max_acc)
                max_acc_round = max_acc_index + 1
                
                line_color = line.get_color()
                
                # Draw a vertical dashed line at the round of max accuracy
                axes[0].axvline(x=max_acc_round, color=line_color, linestyle='--', linewidth=1.5, alpha=0.7)
                
                # Position text annotation to avoid overlap
                if i % 2 == 0:
                    y_position = max_acc - 0.015 
                    vertical_align = 'top'
                else:
                    y_position = max_acc + 0.005
                    vertical_align = 'bottom'

                # Add a text label showing the max accuracy value
                axes[0].text(
                    x=max_acc_round + 15,
                    y=y_position,
                    s=f'{max_acc:.4f}',
                    color='white',
                    backgroundcolor=line_color,
                    ha='left',
                    va=vertical_align,
                    fontsize=9,
                    fontweight='bold'
                )

            # Plot loss curve
            loss_curve = res['server']['train_loss']
            if loss_curve:
                rounds = np.arange(1, len(loss_curve) + 1)
                axes[1].plot(rounds, np.array(loss_curve), label=label, alpha=1, linewidth=2)

        # Configure plot aesthetics
        for i, ax in enumerate(axes):
            ax.set_xlabel('# of Communication Rounds', size=12)
            if i == 0:
                ax.set_ylabel('Testing Accuracy', size=12)
            if i == 1:
                ax.set_ylabel('Training Loss', size=12)
            ax.legend(prop={'size': 12})
            ax.tick_params(axis='both', labelsize=12)
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
