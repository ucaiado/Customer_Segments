#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Load and clean the data used in the project

@author: ucaiado

Created on 05/26/2016
"""
# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn import decomposition


def load():
    '''
    Load the wholesale customers dataset, transform and exclude selected
    outliers. Return a dataframe of the treated data, another with a
    transformed sample, another with pca transformed and a last one with the
    samples pca transformed
    '''
    # Load the wholesale customers dataset
    data = pd.read_csv('customers.csv')
    data.drop(['Region', 'Channel'], axis=1, inplace=True)

    # Select three indices of your choice you wish to sample from the dataset
    indices = [1, 271, 413]

    # Create a DataFrame of the chosen samples
    samples = pd.DataFrame(data.loc[indices],
                           columns=data.keys()).reset_index(drop=True)
    samples.index = indices

    # Scale the data using the natural logarithm
    log_data = np.log(data.copy())

    # Scale the sample data using the natural logarithm
    log_samples = np.log(samples.copy())

    # OPTIONAL: Select the indices for data points you wish to remove
    outliers = [154, 128, 75, 66]

    # Remove the outliers, if any were specified
    good_data = log_data.drop(log_data.index[outliers]).reset_index(drop=True)

    # Fit PCA to the good data using only two dimensions
    pca = decomposition.PCA(n_components=2)
    pca.fit(good_data)

    # Apply a PCA transformation the good data
    reduced_data = pca.transform(good_data)

    # Apply a PCA transformation to the sample log-data
    pca_samples = pca.transform(log_samples)

    # Create a DataFrame for the reduced data
    reduced_data = pd.DataFrame(reduced_data,
                                columns=['Dimension 1', 'Dimension 2'])

    return good_data, log_samples, reduced_data, pca_samples
