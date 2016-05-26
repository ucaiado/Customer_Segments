#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Library to explore the the dataset used

@author: ucaiado

Created on 05/16/2016
"""
# load required libraries
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def features_boxplot(all_data, samples, indices):
    '''
    Return a matplotlib object with a boxplot of all features in the dataset
    pointing out the data in the sample using dots. The legend is the indices
    of the sample in all_data
    :param all_data: DataFrame. all dataset with 6 columns
    :param samples: DataFrame. a sample of all data with 6 columns and 3 lines
    :param indices: list. the original indices of the sample in all_data
    '''
    # reshape the datasets
    data2 = pd.DataFrame(all_data.stack())
    data2.columns = ['annual spending']
    data2.index.names = ['IDX', 'Product']
    data2.reset_index(inplace=True)
    # reshape the sample
    samples2 = samples.copy()
    samples2.index = indices
    samples2 = pd.DataFrame(samples2.stack())
    samples2.columns = ['annual spending']
    samples2.index.names = ['IDX', 'Product']
    samples2.reset_index(inplace=True)
    samples2.index = samples2.IDX
    # Plot the  annual spending with horizontal boxes
    ax = sns.boxplot(x='annual spending', y='Product', data=data2,
                     whis=np.inf, color='lightgrey')
    # Add in points to show each selected observation
    sns.stripplot(x='annual spending',
                  y='Product',
                  data=samples2,
                  hue='IDX',
                  size=9,
                  palette=sns.color_palette('Set2', 5),
                  linewidth=0)
    # insert a title
    ax.set_title('Annual Spending in Monetary Units by Product',
                 fontsize=16, y=1.03)

    return ax


def pca_results(good_data, pca):
    '''
    Create a DataFrame of the PCA results. Includes dimension feature weights
    and explained variance Visualizes the PCA results
    :param good_data: DataFrame. all dataset log transformed with 6 columns
    :param pca: Sklearn Object. a PCA decomposition object already fitted
    '''
    # Dimension indexing
    dimensions = dimensions = ['Dimension {}'.format(i)
                               for i in range(1, len(pca.components_)+1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4),
                              columns=good_data.keys())
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4),
                                   columns=['Explained Variance'])
    variance_ratios.index = dimensions

    # reshape the data to be plotted
    df_aux = components.unstack().reset_index()
    df_aux.columns = ["Feature", "Dimension", "Variance"]

    # Create a bar plot visualization
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot the feature weights as a function of the components
    sns.barplot(x="Dimension", y="Variance", hue="Feature", data=df_aux, ax=ax)
    ax.set_ylabel("Feature Weights")
    ax.set_xlabel("")
    ax.set_xticklabels(dimensions, rotation=0)

    # Display the explained variance ratios
    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05,
                "Explained Variance\n          %.4f" % (ev))

    # insert a title
    # ax.set_title('PCA Explained Variance Ratio',
    #              fontsize=16, y=1.10)

    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis=1)
