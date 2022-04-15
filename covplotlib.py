
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

def plot_ts(climate_df, varnames):
    print(climate_df)
    nrows = 3
    ncols = 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(12,8))

    for i in range(len(varnames)):
        ax[int(np.floor(i/ncols)), int(np.mod(i,ncols))].plot(climate_df['date'], climate_df[varnames[i]], c='k', linewidth=0.75)
        ax[int(np.floor(i/ncols)), int(np.mod(i,ncols))].axvline(climate_df['date'].values[-1], c='crimson', linestyle='--', linewidth=1.25)
        ax[int(np.floor(i/ncols)), int(np.mod(i,ncols))].set_title(varnames[i])

    plt.tight_layout()
    plt.show()
    return
