#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     streamlit_indoorLoc_visualizer.py
# @author   Zhengyu Ma
#
# @brief    A visualization tool for indoor localization system based on streamlit
#           web interface and DNN-based scalable indoor localization scheme [1].
#
# @remarks  References
#           [1] Kyeong Soo Kim, Sanghyuk Lee, and Kaizhu Huang,
#               "A scalable deep neural network architecture for multi-building
#               and multi-floor indoor localization based on Wi-Fi fingerprinting,"
#               Big Data Analytics, vol. 3, no. 4, pp. 1-17, Apr. 19, 2018.
#               Available online: https://doi.org/10.1186/s41044-018-0031-2
#

# import modules
import math
import numpy as np
import pandas as pd
import random
import streamlit as st
import plotly.graph_objs as go
from sklearn.preprocessing import scale
#from tensorflow.keras.models import load_model
from tensorflow.python.keras.models import load_model

# Set page config
st.set_page_config(
    page_title="IndoorLoc Visualizer",
    page_icon="U+1F4CD",
    layout="wide",
)

# Set up the app sidebar
st.title(':green[_Visualization tool for Wi-Fi Fingerprinting Indoor Localization_]')
st.write("##### You could check out the visualizer and this web app first to familiar with,"
         " then select a row number, here are some tips for you")
st.write("- Sidebar is used for pages and input methods, you may collapse it. Also you can change settings to wide mode")
st.write("- The visualizer has its tools on the right, choose them to help your operation")
st.write("- Mouse scroll used for zooming, shift + holding click used for view angle spinning,"
         " upper right is full screen mode for better view")
st.write("- Below the visualizer is comparison info between data row you chose and estimated "
         "position in two algorithms")
st.write("- visualizer will show two traces of algorithms in different colors, red stands for mean,"
         " blue stands for weighted")
with st.sidebar:
    #st.image('/Users/jackma/Documents/2YEAR4-S1/FYP/Web_interface/streamlit_learn/img_04.png')
    st.title(':green[_Scalable indoor localization model test_]')
    st.info('This application is for testing of trained models with direct interaction and results display')

# define global constants
project_home = '/Users/jackma/Documents/2YEAR4-S1/FYP/Web_interface/indoor_localization_prototype-main'
path_train = './UJIIndoorLoc/trainingData2.csv'  # '-110' for the lack of AP.
path_validation = './UJIIndoorLoc/validationData2.csv'  # ditto
path_model = './results/scalable_indoor_localization_model'
N = 8
scaling = 0.2
training_ratio = 0.9

# read both train and test dataframes for consistent label formation through one-hot encoding
train_df = pd.read_csv(path_train, header=0)  # pass header=0 to be able to replace existing names
test_df = pd.read_csv(path_validation, header=0)

train_AP_features = scale(np.asarray(train_df.iloc[:, 0:520]).astype(float),
                          axis=1)  # convert integer to float and scale jointly (axis=1)
train_df['REFPOINT'] = train_df.apply(lambda row: str(int(row['SPACEID'])) + str(int(row['RELATIVEPOSITION'])),
                                      axis=1)  # add a new column

blds = np.unique(train_df[['BUILDINGID']])
flrs = np.unique(train_df[['FLOOR']])
x_avg = {}
y_avg = {}
for bld in blds:
    for flr in flrs:
        # map reference points to sequential IDs per building-floor before building labels
        cond = (train_df['BUILDINGID'] == bld) & (train_df['FLOOR'] == flr)
        _, idx = np.unique(train_df.loc[cond, 'REFPOINT'], return_inverse=True)  # refer to numpy.unique manual
        train_df.loc[cond, 'REFPOINT'] = idx

        # calculate the average coordinates of each building/floor
        x_avg[str(bld) + '-' + str(flr)] = np.mean(train_df.loc[cond, 'LONGITUDE'])
        y_avg[str(bld) + '-' + str(flr)] = np.mean(train_df.loc[cond, 'LATITUDE'])

# build labels for multi-label classification
len_train = len(train_df)
blds_all = np.asarray(pd.get_dummies(pd.concat(
    [train_df['BUILDINGID'], test_df['BUILDINGID']])))  # for consistency in one-hot encoding for both dataframes
flrs_all = np.asarray(pd.get_dummies(pd.concat([train_df['FLOOR'], test_df['FLOOR']])))  # ditto
blds = blds_all[:len_train]
flrs = flrs_all[:len_train]
rfps = np.asarray(pd.get_dummies(train_df['REFPOINT']))
train_labels = np.concatenate((blds, flrs, rfps), axis=1)

# split the training set into training and validation sets; we will use the
# validation set at a testing set.
train_val_split = np.random.rand(len(train_AP_features)) < training_ratio  # mask index array
x_train = train_AP_features[train_val_split]
y_train = train_labels[train_val_split]
x_val = train_AP_features[~train_val_split]
y_val = train_labels[~train_val_split]

### build and train a complete model with the trained SAE encoder and a new classifier
# st.write("\nLoading the saved model ...\n")
model = load_model(path_model)

# turn the given validation set into a testing set
test_AP_features = scale(np.asarray(test_df.iloc[:, 0:520]).astype(float),
                         axis=1)  # convert integer to float and scale jointly (axis=1)
x_test_utm = np.asarray(test_df['LONGITUDE'])
y_test_utm = np.asarray(test_df['LATITUDE'])
blds = blds_all[len_train:]
flrs = flrs_all[len_train:]

# row_number = int(input('Insert a number（the row number of test dataset that you choose）'))
n_rows = test_df.shape[0]
row_number = st.slider(':violet[Pick a Row from TestSet as input within range of 1110, '
                       'the testing results will show below]',
                       min_value=0,
                       max_value=n_rows-1)
# row_number = st.number_input('Input a row number',
#                              min_value=0,
#                              max_value=n_rows-1)
test_row = test_df.iloc[[row_number]]
test_rss = test_AP_features[[row_number]]

# calculate the accuracy of building and floor estimation
preds = model(test_rss, training=False)[0].numpy()

# calculate positioning error when building and floor are correctly estimated
x = 0.0
x_weighted = 0.0
y = 0.0
y_weighted = 0.0
pos_err = -1.0  # initial value as an indicator of no processing
pos_err_weighted = -1.0  # ditto
if test_row['BUILDINGID'].values[0] == np.argmax(preds[:3]) and test_row['FLOOR'].values[0] == np.argmax(preds[3:8]):
    x_test_utm = x_test_utm[row_number]
    y_test_utm = y_test_utm[row_number]
    blds = blds[row_number]
    flrs = flrs[row_number]
    rfps = preds[8:118]
    idxs = np.argpartition(rfps, -N)[-N:]  # (unsorted) indexes of up to N nearest neighbors
    threshold = scaling * np.amax(rfps)
    xs = []
    ys = []
    ws = []
    for i in idxs:
        rfp = np.zeros(110)
        rfp[i] = 1
        rows = np.where((train_labels == np.concatenate((blds, flrs, rfp))).all(axis=1))[0]
        if rows.size > 0:
            if rfps[i] >= threshold:
                xs.append(train_df.loc[train_df.index[rows[0]], 'LONGITUDE'])
                ys.append(train_df.loc[train_df.index[rows[0]], 'LATITUDE'])
                ws.append(rfps[i])
    if len(xs) > 0:
        x = np.mean(xs)
        y = np.mean(ys)
        pos_err = math.sqrt((x - x_test_utm) ** 2 + (y - y_test_utm) ** 2)
        x_weighted = np.average(xs, weights=ws)
        y_weighted = np.average(ys, weights=ws)
        pos_err_weighted = math.sqrt((x_weighted - x_test_utm) ** 2 + (y_weighted - y_test_utm) ** 2)
    else:
        key = str(np.argmax(blds)) + '-' + str(np.argmax(flrs))
        x = x_weighted = x_avg[key]
        y = y_weighted = y_avg[key]
        pos_err = pos_err_weighted = math.sqrt((x - x_test_utm) ** 2 + (y - y_test_utm) ** 2)

### display input and output
st.write('_Row number:_\t', row_number)
if pos_err < 0:
    st.warning(":red[failed to estimated by this data, please change to another row number]")

#there are some data test fail 905, 348...
#do we need to show weighted coordinates they are the same
#sae model doesn't work
#input as text input because slider can not accurately point to a specific row number

# Create 3D scatter plot using Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=train_df["LONGITUDE"],
    y=train_df["LATITUDE"],
    z=train_df["FLOOR"],
    mode="markers",
    marker=dict(
        size=4,
        color=train_df["FLOOR"],
        colorscale="Agsunset",
        opacity=1
    )
)])

# Add estimated points to the center of the plot
if row_number != 0:
    fig.add_trace(go.Scatter3d(
        x=[x],
        y=[y],
        z=[np.argmax(preds[3:8])],
        mode="markers+text",
        text="Estimated_position",
        marker=dict(
            size=10,
            color="red",
            opacity=1.0,
        )
    ))

    fig.add_trace(go.Scatter3d(
        x=[x_weighted],
        y=[y_weighted],
        z=[np.argmax(preds[3:8])],
        mode="markers",
        marker=dict(
            size=10,
            color="blue",
            opacity=0.5,
        )
    ))

# Set layout of the plot
fig.update_layout(
    autosize=True,
    title="3D Scatter of dataset",
    #height=400,
    margin=dict(t=25, r=0, l=0, b=0),
    showlegend=False,
    modebar_orientation="v",
    hoverdistance=10,
    scene=dict(
        xaxis=dict(range=[-7700, -7300], autorange=True, title="Longitude"),
        yaxis=dict(range=[4864750, 4865000], autorange=True, title="Latitude"),
        zaxis=dict(range=[-1, 4], autorange=False, title="Floor"),
        aspectratio=dict(x=1.62, y=1.2, z=0.5),
        camera=dict(
            up=dict(x=0, y=0, z=1),
            eye=dict(x=-0.44, y=-1.4, z=0.5)
        )
    )
)
#reset the camera to estimated location

locating = "Allocated to Estimated position"

if row_number != 0:
    fig.update_layout(title=locating,
                      )

# Show Plotly scatter plot using Streamlit
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.write('### True Position')
    st.write('- Building ID:\t', test_row['BUILDINGID'].values[0])
    st.write('- Floor ID:\t', test_row['FLOOR'].values[0])
    st.write('#### :green[Coordinates]')
    st.write('- X:\t', test_row['LONGITUDE'].values[0])
    st.write('- Y:\t', test_row['LATITUDE'].values[0])


with col2:
    if pos_err >= 0:
        st.write('### Estimated Position')
        st.write('- Building ID:\t', np.argmax(preds[:3]))
        st.write('- Floor ID:\t', np.argmax(preds[3:8]))
        st.write('#### :green[Coordinates]')
        st.write('- X:\t', x)
        st.write('- Y:\t', y)
        st.write('- Positioning error :\t', pos_err)
        st.write('#### :green[Coordinates (weighted)]')
        st.write('- X:\t', x_weighted)
        st.write('- Y:\t', y_weighted)
        st.write('- Positioning error :\t', pos_err_weighted)
    else:
        st.write('### Estimated Position')
        st.write('- :red[Building/Floor estimation failed!]')
        st.write('- :red[Try to slide the bar]')

