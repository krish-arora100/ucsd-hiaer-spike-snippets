import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-out-dir",
    default="/home/k7arora/output/misc",
    type=str,
    help="dir path that stores the trained model checkpoint",
)
args = parser.parse_args()
output_dir = args.out_dir

def plot_membrane_potentials_fn(output_dir):
    # List of (layer_type, layer_idx, neuron_idx) to plot
    plot_targets = [
        ("conv", 0, 0), ("conv", 0, 1), ("conv", 0, 2),
        ("fc", 0, 0), ("fc", 0, 1), ("fc", 0, 2),
        ("output", 0, 0), ("output", 0, 1), ("output", 0, 2),
    ]
    # Map layer_type to index in thresholds list
    layer_type_to_idx = {"conv": 0, "fc": 1, "output": 2}
    debug = True
    # Load thresholds
    thresh_path = os.path.join(output_dir, f"thresholds.npy")
    if os.path.exists(thresh_path):
        thresholds = np.load(thresh_path, allow_pickle=True)
    else:
        thresholds = None
    for layer_type, layer_idx, neuron_idx in plot_targets:
        # File naming convention
        sj_path = os.path.join(output_dir, f"potentials_spikingjelly_{layer_type}{layer_idx}_neuron{neuron_idx}.npy")
        conv_path = os.path.join(output_dir, f"potentials_converted_{layer_type}{layer_idx}_neuron{neuron_idx}.npy")
        if not (os.path.exists(sj_path) and os.path.exists(conv_path)):
            print(f"Missing: {sj_path} or {conv_path}")
            continue
        v_sj = np.load(sj_path, allow_pickle=True)
        v_conv = np.load(conv_path, allow_pickle=True)
        # Replace None or nan with 0 for plotting
        def clean_trace(trace):
            arr = np.array(trace, dtype=float)
            arr[np.isnan(arr)] = 0.0
            if arr.dtype == object:
                arr = np.array([0.0 if (x is None or (isinstance(x, float) and np.isnan(x))) else x for x in arr], dtype=float)
            return arr
        v_sj = clean_trace(v_sj)
        v_conv = clean_trace(v_conv)
        timesteps = np.arange(len(v_sj))
        if debug:
            print(f"Plotting membrane potentials for {layer_type}{layer_idx} neuron {neuron_idx}")
            print(f"SpikingJelly: {sj_path} - shape: {v_sj.shape}")
            print(f"Converted SNN: {conv_path} - shape: {v_conv.shape}")
            print(f"Timesteps: {timesteps.shape}")
            print(f"SpikingJelly Membrane Potential: {v_sj[:10]}...")  # Print first 10 values
            print(f"Converted Membrane Potential: {v_conv[:10]}...")  #

            debug = True

        plt.figure(figsize=(8,4))
        plt.plot(timesteps, v_sj, label='SpikingJelly', color='blue')
        plt.plot(timesteps, v_conv, label='Converted SNN', color='red')
        # Add thin, dotted black vertical lines every 11 timesteps (image boundaries)
        n_steps = len(timesteps)
        image_period = 11
        for x in range(image_period, n_steps, image_period):
            plt.axvline(x=x, color='black', linestyle=':', linewidth=0.7, zorder=1)
        # Add threshold line and spike dots
        threshold = None
        if thresholds is not None:
            threshold = thresholds[layer_type_to_idx[layer_type]]
            plt.axhline(y=threshold, color='black', linestyle='dotted', label='Threshold')
        # Plot spike dots above the traces
        if threshold is not None:
            # Calculate y positions for dots based on max/min of both traces
            v_max = max(np.max(v_sj), np.max(v_conv))
            v_min = min(np.min(v_sj), np.min(v_conv))
            # Blue (SJ) dots: 1.03*max - 0.03*min
            # Red (Converted) dots: 1.06*max - 0.06*min
            sj_spikes = np.where(v_sj >= threshold)[0]
            if len(sj_spikes) > 0:
                ydot_sj = 1.03 * v_max - 0.03 * v_min
                plt.scatter(sj_spikes, [ydot_sj]*len(sj_spikes), color='blue', marker='o', s=6, label='SJ Spike', zorder=5)
            conv_spikes = np.where(v_conv >= threshold)[0]
            if len(conv_spikes) > 0:
                ydot_conv = 1.06 * v_max - 0.06 * v_min
                plt.scatter(conv_spikes, [ydot_conv]*len(conv_spikes), color='red', marker='o', s=6, label='Converted Spike', zorder=5)
        plt.xlabel('Time step')
        plt.ylabel('Membrane potential')
        plt.title(f'Membrane Potential: {layer_type}{layer_idx} neuron {neuron_idx}')
        plt.legend()
        plt.tight_layout()
        plot_dir = output_dir + "/plots"
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, f"membrane_potential_{layer_type}{layer_idx}_neuron{neuron_idx}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")


plot_membrane_potentials_fn(output_dir)
