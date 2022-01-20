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
def _show_interactive_plot(
    df,
    columns,
    show_function,
    widget_dic={},
    ):

    widget_dic = {k: v for k, v in widget_dic.items() if v is not None}
    widget_list = [w for k, w in widget_dic.items()]

    ui = widgets.HBox(widget_list)

    out = widgets.interactive_output(
        show_function,
        widget_dic,
    )
    display(ui, out)


def show_histogram(
    df,
    columns=None,
    **kwargs,
    ):
    '''Show an interactive histogram of a dataframe.'''
    if columns is None:
        columns = list(df.columns)
    
    # Widgets for interacting with the histogram
    select_column = widgets.Select(
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

    # Create and show the histogram
    def _plot_histogram(column, bins):
        sns.displot(
            data=df,
            x=column,
            bins=bins,
            **kwargs,
            )
        plt.show()
    
    _show_interactive_plot(
        df=df,
        columns=columns,
        show_function=_plot_histogram,
        widget_dic={
            'column': select_column,
            'bins': slider_bins,
            },
    )


def show_index_scatterplot(
    df,
    columns=None,
    **kwargs,    
):
    if columns is None:
        columns = list(df.columns)
    
    select_y = widgets.Select(
        options=list(columns),
        value=columns[0],
        description='Feature:',
        disabled=False,
    )

    def _plot_index_scatterplot(y):
        sns.scatterplot(
            data=df.reset_index(),
            x='index',
            y=y,
            **kwargs,
        )
        plt.show()
    
    _show_interactive_plot(df, _plot_index_scatterplot, {'y': select_y})

def show_scatterplot(
    df,
    columns=None,
    **kwargs,    
):
    if columns is None:
        columns = list(df.columns)
    
    select_x = widgets.Select(
        options=list(columns),
        value=columns[0],
        description='x:',
        disabled=False,
    )
    select_y = widgets.Select(
        options=list(columns),
        value=columns[0],
        description='y:',
        disabled=False,
    )

    def _plot_scatterplot(x, y):
        sns.scatterplot(
            data=df,
            x=x,
            y=y,
            **kwargs,
        )
        plt.show()
    _show_interactive_plot(
        df,
        columns,
        _plot_scatterplot,
        {'x': select_x, 'y': select_y},
        )


# %%
