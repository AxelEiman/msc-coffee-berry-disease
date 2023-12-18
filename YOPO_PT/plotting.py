from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

from ultralytics.yolo.utils import LOGGER, plt_settings

@plt_settings()
def plot_results(file='path/to/results.csv', dir='', segment=False, pose=False, classify=False, on_plot=None):
    """Plot training results.csv. Usage: from utils.plots import *; plot_results('path/to/results.csv')."""
    import pandas as pd
    save_dir = Path(file).parent if file else Path(dir)
    if classify:
        fig, ax = plt.subplots(2, 2, figsize=(6, 6), tight_layout=True)
        index = [1, 4, 2, 3]
    elif segment:
        fig, ax = plt.subplots(2, 8, figsize=(18, 6), tight_layout=True)
        index = [1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 15, 16, 7, 8, 11, 12]
    elif pose:
        fig, ax = plt.subplots(2, 9, figsize=(21, 6), tight_layout=True)
        index = [1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16, 17, 18, 8, 9, 12, 13]
    else:
        # CBD change, added two more plots for the MIL losses
        fig, ax = plt.subplots(2, 6, figsize=(12, 6), tight_layout=True)
        index = [1, 2, 3, 4, 6, 7, 10, 11, 12, 5, 8, 9]
    ax = ax.ravel()
    files = list(save_dir.glob('results*.csv'))
    assert len(files), f'No results.csv files found in {save_dir.resolve()}, nothing to plot.'
    for f in files:
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate(index):
                y = data.values[:, j].astype('float')
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker='.', label=f.stem, linewidth=2, markersize=8)  # actual results
                ax[i].plot(x, gaussian_filter1d(y, sigma=3), ':', label='smooth', linewidth=2)  # smoothing line
                ax[i].set_title(s[j], fontsize=12)
                # if j in [8, 9, 10]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            LOGGER.warning(f'WARNING: Plotting error for {f}: {e}')
    ax[1].legend()
    fname = save_dir / 'results.png'
    fig.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)

if __name__ == '__main__':
    plot_results('/home/eiman/data/outputs/detect/mc_pt_10_l1_0_l2_1/results.csv')