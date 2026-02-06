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
    
    # Output-mappe
    outdir = os.path.join("figures", "pinn")
    os.makedirs(outdir, exist_ok=True)

    out = generate_training_data(cfg)

    sensor_data = None
    if isinstance(out, (tuple, list)):
        for v in out:
            if hasattr(v, "ndim") and v.ndim == 2 and v.shape[1] == 4:
                sensor_data = v
                break

    # 2) Tren PINN (MERK: rekkefølge sensor_data først, cfg etterpå)
    pinn_params, losses = train_pinn(sensor_data, cfg)

    # 3) Plot tapskurver
    plt.figure()
    plt.semilogy(np.asarray(losses["total"]), label="total")
    plt.semilogy(np.asarray(losses["data"]), label="data")
    plt.semilogy(np.asarray(losses["physics"]), label="physics")
    plt.semilogy(np.asarray(losses["ic"]), label="ic")
    plt.semilogy(np.asarray(losses["bc"]), label="bc")
    plt.legend()
    plt.title("PINN training losses")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss.png"), dpi=200)
    plt.close()

    # 4) Prediksjon på grid
    # (predict_grid hos dere tar typisk (params["nn"], cfg) eller (params, cfg).
    # Siden data_loss/ic_loss bruker pinn_params["nn"], antar vi det er nn-delen som går inn i predict_grid.)
    # Lag grid fra config
    x = jnp.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    y = jnp.linspace(cfg.y_min, cfg.y_max, cfg.ny)
    t = jnp.linspace(cfg.t_min, cfg.t_max, cfg.nt)

    T_pred = predict_grid(pinn_params["nn"], x, y, t, cfg)

    T_pred = np.asarray(T_pred)

    # 5) Visualisering
    plot_snapshots(x, y, t, T_pred, save_path="figures/snapshotsWIP.png")

    create_animation(x, y, t, T_pred, save_path="figures/animWIP.png") 

    # 6) Print lærte fysiske parametre (hvis dere har dem i pinn_params)
    print("\nLearned physical parameters (if present):")
    for name in ["log_alpha", "log_k", "log_h", "log_P", "alpha", "k", "h", "P"]:
        if name in pinn_params: 
            val = pinn_params[name]
            # hvis log-param:
            if name.startswith("log_"):
                val = jnp.exp(val)
            print(f"{name.replace('log_', '')} = {float(val[0]):.6g}")



    #######################################################################
    # Oppgave 5.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()

