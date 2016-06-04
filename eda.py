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


def cluster_results(reduced_data, preds, centers, pca_samples):
    '''
    Visualizes the PCA-reduced cluster data in two dimensions
    Adds cues for cluster centers and student-selected sample data
    :param reduced_data: pandas dataframe. the dataset transformed and cleaned
    :param preds: numpy array. teh cluster classification of each datapoint
    :param centers: numpy array. the center of the clusters
    :param pca_samples: numpy array. the sample choosen
    '''

    predictions = pd.DataFrame(preds, columns=['Cluster'])
    plot_data = pd.concat([predictions, reduced_data], axis=1)

    # Generate the cluster plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color map
    cmap = sns.color_palette('Set2', 11)

    # Color the points based on assigned cluster
    for i, cluster in plot_data.groupby('Cluster'):
        cluster.plot(ax=ax, kind='scatter', x='Dimension 1', y='Dimension 2',
                     color=cmap[i],
                     label='Cluster %i' % (i),
                     s=30)

    # Plot centers with indicators
    for i, c in enumerate(centers):
        ax.scatter(x=c[0], y=c[1], color='white', edgecolors='black',
                   alpha=1, linewidth=2, marker='o', s=200)
        ax.scatter(x=c[0], y=c[1], marker='$%d$' % (i), alpha=1, s=100)

    # Plot transformed sample points
    ax.scatter(x=pca_samples[:, 0], y=pca_samples[:, 1],
               s=150, linewidth=4, color='black', marker='x')

    # Set plot title
    s_title = 'Cluster Learning on PCA-Reduced Data - Centroids Marked by'
    s_title += ' Number\nTransformed Sample Data Marked by Black Cross\n'
    ax.set_title(s_title, fontsize=16)


def channel_results(reduced_data, outliers, pca_samples, na_index):
    '''
    Visualizes the PCA-reduced cluster data in two dimensions using the full
    dataset Data is labeled by "Channel" and cues added for student-selected
    sample data
    :param reduced_data: pandas dataframe. the dataset transformed and cleaned
    :param outliers: list. the datapoint considered outliers
    :param pca_samples: numpy array. the sample choosen
    :param pca_samples: numpy array. the original IDs of the sample
    '''
    # Check that the dataset is loadable
    try:
        full_data = pd.read_csv('customers.csv')
    except:
        print 'Dataset could not be loaded. Is the file missing?'
        return False

    # Create the Channel DataFrame
    channel = pd.DataFrame(full_data['Channel'], columns=['Channel'])
    channel = channel.drop(channel.index[outliers]).reset_index(drop=True)
    labeled = pd.concat([reduced_data, channel], axis=1)

    # Generate the cluster plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color map
    # cmap = cm.get_cmap('gist_rainbow')
    cmap = sns.color_palette('Set2', 11)

    # Color the points based on assigned Channel
    labels = ['Hotel/Restaurant/Cafe', 'Retailer']
    grouped = labeled.groupby('Channel')
    for i, channel in grouped:
        channel.plot(ax=ax, kind='scatter', x='Dimension 1', y='Dimension 2',
                     color=cmap[i],
                     label=labels[i-1],
                     s=30)

    # Plot transformed sample points
    for i, sample in zip(na_index, pca_samples):
        ax.scatter(x=sample[0], y=sample[1], s=200,
                   linewidth=3,
                   color='black',
                   marker='o',
                   facecolors='none')
        ax.scatter(x=sample[0]+0.25, y=sample[1]+0.3,
                   marker='$%d$' % (i),
                   alpha=1,
                   s=125)

    # Set plot title
    s_title = 'PCA-Reduced Data Labeled by "Channel"\nTransformed Sample Data'
    s_title += ' Circled'
    ax.set_title(s_title)


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
