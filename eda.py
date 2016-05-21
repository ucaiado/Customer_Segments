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
                     whis=np.inf, color='c')
    # Add in points to show each selected observation
    sns.stripplot(x='annual spending', y='Product', data=samples2,
                  hue='IDX', size=9, color='0.3', linewidth=0)
    # insert a title
    ax.set_title('Annual Spending in Monetary Units by Product',
                 fontsize=16, y=1.03)

    return ax
