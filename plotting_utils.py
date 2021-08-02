import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
import seaborn as sns
import matplotlib.dates as mdates
import datetime as dt
from dateutil import tz
from importlib import reload  
from IPython.display import display, HTML, display_html
from itertools import chain,cycle
from operator import itemgetter
import networkx as nx
from networkx.algorithms import community
import math 
import os


#  I like to create a map for whenever I need to calculate pair-wise correlation or make a matrix of density plots
# So then I can calculate correlations, p-values and make plots all in one loop!
metrics_map = []
metrics_map.append({'col':'data_col1', 'label':'Data Column 1', 'log':False, 'scale':None, 'bins':6})
metrics_map.append({'col':'data_col2', 'label':'Data Column 2', 'log':False, 'scale':None, 'bins':50})
metrics_map.append({'col':'data_col3', 'label':'Data Column 3', 'log':True, 'scale':None, 'bins':50})

def get_data_from_map(metrics_df, metrics_map, col_key='col', label_key=''):        
    filtered_metrics_cols = [col_key for x in metrics_map]
    filtered_metrics_labels = [label_key for x in metrics_map]
    filtered_metrics = metrics_df[filtered_metrics_cols].copy()
    filtered_metrics = filtered_metrics.applymap(lambda x: 0.00000000001 if x <= 0 else x) # This is to avoid log-scaling errors in plots
    return filtered_metrics, filtered_metrics_labels



def density_plot_matrix(data_df, style_map=None, export_folder="density_plots/", x_rows=-1, y_rows=-1, fig_w=20, fig_h=20, title_fontsize=40, ticks_fontsize=30, sns_style="white", as_latex=False, latex_graphics_tmplt=None):
    latex_graphics_tmplt = "\\subfloat[]{\\includegraphics[width=0.3\\textwidth]{figures/[img]}}\\hfil" if latex_graphics_tmplt is None else latex_graphics_tmplt
    final_latex = ""
    os.makedirs(export_folder)
    sns.set_style(sns_style)
    if x_rows < 0:
        x_rows = math.ceil(math.sqrt(len(data_df.columns)))
    if y_rows < 0:
        y_rows = math.ceil(math.sqrt(len(data_df.columns)))
    if not as_latex:
        f, axs = plt.subplots(x_rows, y_rows, figsize=(fig_w, fig_h))
    x = 0
    y = 0
    df_columns = list(data_df.columns)
    for y in range(0, y_rows):
        if as_latex:
            final_latex += "\n"
        for x in range(0, x_rows):
            i = x_rows*y + x
            if not as_latex:
                ax = axs[y,x]
            else:
                f, ax = plt.subplots(figsize=(fig_w, fig_h))
            if i >= len(df_columns):
                if not as_latex:
                    f.delaxes(ax)
            else:
                col_name = df_columns[i]
                var = style_map[i]['col'] if style_map is not None else col_name
                label = style_map[i]['label'] if style_map is not None else col_name
                log_scale = style_map[i]['log'] if style_map is not None else False
                scale = style_map[i]['scale'] if style_map is not None else None
                bins = style_map[i]['bins'] if style_map is not None else 30
                # print(var, label, log_scale, bins)
                sns.histplot(data_df, x=var, ax=ax, log_scale=log_scale, bins=bins)
                ax.tick_params(axis='both', which='major', labelsize=ticks_fontsize)
                label = f"Log {label}" if log_scale else label
                ax.set_xlabel(label, fontsize=title_fontsize)
                ax.set_ylabel("Frequency",fontsize=title_fontsize)
                if scale is not None:
                    ax.set_xlim(scale)
                if as_latex:
                    f.subplots_adjust(wspace=0.3, hspace=0.3)
                    filename = col_name+'_density.png'
                    final_latex += latex_graphics_tmplt.replace("[img]", filename)+"\n"
                    f.savefig(export_folder+filename, bbox_inches='tight')
    if not as_latex:
        f.subplots_adjust(wspace=0.3, hspace=0.3)
        filename = 'all_density_plots.png'
        f.savefig(export_folder+filename, bbox_inches='tight')
        display(f.tight_layout())
    else:
        display(f.tight_layout())
        print(final_latex)


#corr_plot(spearman_corr_mat, prefix+"new_spearmanr_corr", filtered_metrics_labels, filtered=False)
def corr_plot(corr_mat, postfix, metrics_corr_labels, filtered=False):
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_mat, dtype=bool),k = 1)
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(25, 20))
    display(len(labels))
    # Generate a custom diverging colormap
    cmap =  sns.diverging_palette(30, 240, l=50, s=100, as_cmap=True)
    if filtered:
        corr_mat = corr_mat[(corr_mat >= 0.3) | (corr_mat <= -0.3)]
    # Draw the heatmap with the mask and correct aspect ratio
    if metrics_corr_labels is not None:
        display(sns.heatmap(corr_mat, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, xticklabels=metrics_corr_labels, yticklabels=metrics_corr_labels,square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax = ax, annot = True, fmt=".3f"))
    else:
        display(sns.heatmap(corr_mat, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax = ax, annot = True, fmt=".3f"))
    f.tight_layout()
    f.savefig(f"plots/corr_matrix_{postfix}{'_filtered' if filtered else ''}.png")
    
    
def corr_to_latex(corr_mat, pval_mat, labels):
    n_cols = len(labels)
    spacing = "\\hspace{0.33em}"
    header = "\\begin{tabular}{@{}l|"+"".join(["l" for x in labels])+"@{}}\n\\toprule"
    header += "{Variable} "+"".join(["& {"+str(i+1)+".} " for i in range(0, len(labels))])+"\\\ \n\\bottomrule\n\n"
    footer = "\n\\bottomrule\n\\end{tabular}\n"
    content = ""
    for idx in range(0, len(labels)):
        col = corr_mat.columns[idx]
        content += f"%{col}\n"
        content += f"{idx+1}. {labels[idx]} & "
        for idx2 in range(0, len(labels)-1):
            corr = corr_mat[col].values[idx2]
            pval = pval_mat[col].values[idx2]
            if idx2 < idx:
                stars = ""
                content += f"{spacing} "
                if pval < 0.01:
                    content += "\\bf"
                    stars = "\\textsuperscript{***}"
                elif pval < 0.05:
                    content += "\\bf"
                    stars = "\\textsuperscript{**}"
                elif pval < 0.1:
                    stars = "\\textsuperscript{*}"
                content += f" {corr:.2f}{stars}"
            elif idx2 == idx:
                content += f"{spacing}---"
            content += " & "
        content += "\\\ \n"
    return header+content+footer
                
        
    n_cols = len(labels)
    spacing = "\\hspace{0.33em}"
    header = "\\begin{tabular}{@{}l|"+"".join(["l" for x in labels])+"@{}}\n\\toprule"
    header += "{Variable} "+"".join(["& {"+str(i+1)+".} " for i in range(0, len(labels))])+"\\\ \n\\bottomrule\n\n"
    footer = "\n\\bottomrule\n\\end{tabular}\n"
    content = ""
    for idx in range(0, len(labels)):
        content += f"{idx+1}. {labels[idx]} & "
        col = corr_mat.columns[idx]
        for idx2 in range(0, len(labels)-1):
            corr = corr_mat[col].values[idx2]
            pval = pval_mat[col].values[idx2]
            if idx2 < idx:
                stars = ""
                content += f"{spacing} "
                if pval < 0.01:
                    content += "\\bf"
                    stars = "\\textsuperscript{***}"
                elif pval < 0.05:
                    content += "\\bf"
                    stars = "\\textsuperscript{**}"
                elif pval < 0.1:
                    stars = "\\textsuperscript{*}"
                content += f" {corr:.2f}{stars}"
            elif idx2 == idx:
                content += f"{spacing}---"
            content += " & "
        content += "\\\ \n"
    print(header+content+footer)
    return header+content+footer
                
        
