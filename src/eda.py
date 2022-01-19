#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import ipywidgets as widgets
from IPython.display import display

#%% df for testing purposes
dataset = load_iris()
df = pd.DataFrame(
    dataset.data,
    columns=dataset.feature_names,
    )
y = dataset.target
df['y'] = y

# %%
def show_histogram(df, columns=None):
    '''Show an interactive histogram of a dataframe.'''
    if columns is None:
        columns = list(df.columns)
    
    # Widgets for interacting with the histogram
    cbx_cols = widgets.Combobox(
        options=list(columns),
        value=columns[0],
        description='Columns:',
        disabled=False,
    )
    slider_bins = widgets.IntSlider(
        min=1,
        max=100,
        step=1,
        value=10,
        description='Bins:',
        disabled=False,
    )

    ui = widgets.HBox([cbx_cols, slider_bins])

    # Create and show the histogram
    def _plot_histogram(column, bins):
        sns.histplot(data=df, x=column, bins=bins)
        plt.show()

    out = widgets.interactive_output(
        _plot_histogram,
        {
            'column': cbx_cols,
            'bins': slider_bins,
        }
    )
    display(ui, out)
