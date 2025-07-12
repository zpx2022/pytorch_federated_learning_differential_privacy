#!/usr/bin/env python3
import matplotlib.pyplot as plt
import argparse
import os
from recorder import Recorder

def fed_args():
    """
    Defines command-line arguments for the evaluation script.
    :return: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-rr', '--sys-res_root', type=str, required=True, help='Root directory of the results')
    args = parser.parse_args()
    return args

def res_eval():
    """
    Main function for result evaluation.
    It loads all result files from a directory and plots them.
    """
    args = fed_args()
    recorder = Recorder()

    res_files = [f for f in os.listdir(args.sys_res_root)]
    for f in res_files:
        if f.endswith('.json'): # Ensure only json files are loaded
            recorder.load(os.path.join(args.sys_res_root, f), label=f)
    
    recorder.plot()
    plt.show()

if __name__ == "__main__":
    res_eval()
