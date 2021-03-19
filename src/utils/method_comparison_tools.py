# ************************************************************************* #
#     Author:   Pawel Rosikiewicz                                           #       
#     Copyrith: IT IS NOT ALLOWED TO COPY OR TO DISTRIBUTE                  #
#               these file without written                                  #
#               persmission of the Author                                   #
#     Contact:  prosikiewicz@gmail.com                                      #
#                                                                           #
# ************************************************************************* #


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os # allow changing, and navigating files and folders, 
import sys
import re # module to use regular expressions, 
import glob # lists names in folders that match Unix shell patterns
import random # functions that use and generate random numbers

import cv2
import numpy as np # support for multi-dimensional arrays and matrices
import pandas as pd # library for data manipulation and analysis
import seaborn as sns # advance plots, for statistics, 
import matplotlib as mpl # to get some basif functions, heping with plot mnaking 
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt # for making plots, 

from PIL import Image, ImageDraw
import matplotlib.gridspec
from scipy.spatial import distance
from scipy.cluster import hierarchy
from matplotlib.font_manager import FontProperties
from scipy.cluster.hierarchy import leaves_list, ClusterNode, leaders
from sklearn.metrics import accuracy_score

import graphviz # allows visualizing decision trees,
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier # accepts only numerical data
from sklearn.tree import export_graphviz
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#. require for plots below, 
from src.utils.image_augmentation import * # to create batch_labels files, 
from src.utils.data_loaders import load_encoded_imgbatch_using_logfile, load_raw_img_batch
from src.utils.tools_for_plots import create_class_colors_dict
from src.utils.example_plots_after_clustering import plot_img_examples, create_spaces_between_img_clusters, plot_img_examples_from_dendrogram
from src.utils.annotated_pie_charts import annotated_pie_chart_with_class_and_group, prepare_img_classname_and_groupname








# Function, ..............................................................................
def boxplot_with_acc_from_different_models(*, title="", summary_df, dtype, 
                                           figsize=(10,4), legend__bbox_to_anchor=(0.5, 1.3), 
                                           cmap="tab10", cmap_colors_from=0, cmap_colors_to=1, 
                                           legend_ncols=1):
    """
        Small helper function to crreate nice boxplot with the data 
        provided by summary df, after exploring many different models
        
        # inputs
        dtype       : str {"test", "train", "valid"}
        summary_df  : summary dataframe with accuracy and parameter 
                      results returned by grid search developed for all models 
        figsize.    : tuple, two integers, eg: (10, 5)
                      
        # returns, 
        boxplot     : 
        
        summary_df=summary_df_for_boxplot
        figsize=(10,4)
        dtype="valid"
        t = True
        if t==True:
        
    """
    
    # ...............................................
    # data preparation

    # find all modules
    module_names = summary_df.module.unique().tolist()

    # get results for each method, 
    bx_data = list() # list with arr's with values for each box,  
    bx_labels = list() # on x-axis, for each box, 
    bx_modules = list() # set as different colors, 
    # ..
    for module_name in module_names:
        one_module_summary_df = summary_df.loc[summary_df.module==module_name,:]
        # ..
        for one_method in one_module_summary_df.method.unique().tolist():

            acc_data = one_module_summary_df.loc[one_module_summary_df.method==one_method,f"model_acc_{dtype}"]
            acc_data = acc_data.dropna()

            if len(acc_data)>0:
                # ...
                bx_data.append(acc_data.values)
                bx_labels.append(one_method)
                bx_modules.append(module_name)
            else:
                pass

    # find memdians and reorder 
    bx_data_medians = list()
    for i, d in enumerate(bx_data):
        bx_data_medians.append(np.median(d))

    # ...
    temp_df = pd.DataFrame({
        "labels": bx_labels,
        "medians": bx_data_medians,
        "modules": bx_modules
    })
    new_order = temp_df.sort_values("medians", ascending=True).index.values
    # ...
    ordered_bx_data = list()
    ordered_bx_labels = list()
    ordered_bx_modules = list()
    for i in new_order:
        ordered_bx_data.append(bx_data[i])
        ordered_bx_labels.append(bx_labels[i])
        ordered_bx_modules.append(bx_modules[i])

    # ...............................................    
    # set colors for different modules,  
    module_colors = create_class_colors_dict(
        list_of_unique_names=module_names, 
        cmap_name=cmap, 
        cmap_colors_from=cmap_colors_from, 
        cmap_colors_to=cmap_colors_to
    )

    # ...............................................
    # boxplot,  - plt.boxplot(ordered_bx_data);
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    fig.suptitle(title, fontsize=20)

    # add boxes,
    bx = ax.boxplot(ordered_bx_data, 
            showfliers=True,                  # remove outliers, because we are interested in a general trend,
            vert=True,                        # boxes are vertical
            labels=ordered_bx_labels,           # x-ticks labels
            patch_artist=True,
            widths=0.3
    )
    ax.grid(ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticklabels(ordered_bx_labels, rotation=45, fontsize=12, ha="right")
    ax.set_yticks([0, .2, .4, .6, .8, 1])
    ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=15)
    ax.set_ylabel("Accuracy\n", fontsize=20)
    ax.set_xlabel("Method", fontsize=20)
    ax.set_ylim(0,1.02)

    # add colors to each box individually,
    
    for i, j in zip(range(len(bx['boxes'])),range(0, len(bx['caps']), 2)) :
        median_color  ="black"
        box_color     = module_colors[ordered_bx_modules[i]]

        # set properties of items with the same number as boxes,
        plt.setp(bx['boxes'][i], color=box_color, facecolor=median_color, linewidth=2, alpha=0.8)
        plt.setp(bx["medians"][i], color=median_color, linewidth=2)
        plt.setp(bx["fliers"][i], markeredgecolor="black", marker=".") # outliers

        # set properties of items with the 2x number of features as boxes,
        plt.setp(bx['caps'][j], color=median_color)
        plt.setp(bx['caps'][j+1], color=median_color)
        plt.setp(bx['whiskers'][j], color=median_color)
        plt.setp(bx['whiskers'][j+1], color=median_color)

        
    # set xtick labels,  
    for i, xtick in enumerate(ax.get_xticklabels()):
        xtick.set_color(module_colors[ordered_bx_modules[i]])

    if len(module_names)>0:

        # create patch for each dataclass, - adapted to even larger number of classes then selected for example images, 
        patch_list_for_legend =[]
        for i, m_name in enumerate(list(module_colors.keys())):
            label_text = f"{m_name}"
            patch_list_for_legend.append(mpl.patches.Patch(color=module_colors[m_name], label=label_text))

        # add patches to plot,
        fig.legend(
            handles=patch_list_for_legend, frameon=False, 
            scatterpoints=1, ncol=legend_ncols, 
            bbox_to_anchor=legend__bbox_to_anchor, fontsize=15)

        # create space for the legend
        fig.subplots_adjust(top=0.8)    

        
    # add line with baseline
    acc_baseline = summary_df.loc[:,f"baseline_acc_{dtype}"].dropna().values.flatten()
    ax.axhline(acc_baseline[0], lw=2, ls="--", color="dimgrey")
    ax.text(len(ordered_bx_data)+0.4, acc_baseline[0]+0.05, "most frequent baseline", ha="right", color="dimgrey", fontsize=15)        
        
        
    # color patches behing boxplots, 
    patch_width = 2
    patch_color = "lightgrey"
    pathces_starting_x = list(range(0, len(ordered_bx_data), patch_width*2))
    # ...
    for i, sx in enumerate(pathces_starting_x):
        rect = plt.Rectangle((sx+0.5, 0), patch_width, 1000, color=patch_color, alpha=0.8, edgecolor=None)
        ax.add_patch(rect)        
        
        
    # color patches for styling the accuracy, 
    rect = plt.Rectangle((0,0), len(ordered_bx_data)*100, acc_baseline[0], color="red", alpha=0.2, edgecolor=None)
    ax.add_patch(rect)          
    rect = plt.Rectangle((0,acc_baseline[0]), len(ordered_bx_data)*100, 0.7-acc_baseline[0], color="orange", alpha=0.2, edgecolor=None)
    ax.add_patch(rect)             
    rect = plt.Rectangle((0,0.7), len(ordered_bx_data)*100, 10, color="forestgreen", alpha=0.2, edgecolor=None)
    ax.add_patch(rect)            
        
        
        


    # ...............................
    return fig



    
    




















