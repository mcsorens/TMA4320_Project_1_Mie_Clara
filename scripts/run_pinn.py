"""Script for training and plotting the PINN model."""

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots

from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_pinn,
)


def main():
    cfg = load_config("config.yaml")

    #######################################################################
    # Oppgave 5.4: Start
    #######################################################################
train_data = generate_training_data(cfg)

try: 
    pinn_params, history = train_pinn(train_data, cfg)
except TypeError:
    pinn_params, history = train_pinn(cfg, train_data)

print("\nLearned physical parameters: ")
if "log_alpha" in pinn_params:
    print(f" alpha = {float(jnp.exp(pinn_params['log_alpha'])):.6g}")
if "log_k" in pinn_params:
    print(f" k = {float(jnp.exp(pinn_params['log_k'])):.6g}")
if "log_h" in pinn_params:
    print(f" h = {float(jnp.exp(pinn_params['log_h'])):.6g}")
if "log_power" in pinn_params:
    print(f" P = {float(jnp.exp(pinn_params['log_power'])):.6g}")

pred = predict_grid(pinn_params, cfg)

if isinstance(pred, tuple):
    T_pred = pred[0]
else:
    T_pred = pred

try:
    plot_snapshots(T_pred, cfg, title= "PINN prediction")
except TypeError:
    plot_snapshots(T_pred, cfg)

try:
    create_animation(T_pred, cfg, title = "PINN prediction")
except TypeError:
    create_animation(T_pred, cfg)

    #######################################################################
    # Oppgave 5.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
