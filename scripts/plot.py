import streamlit as st
import numpy as np
import torch
import plotly.graph_objs as go


if st.checkbox('load data and plot'):
    st.header('Average displacement error between real and generated trajectory')
    d = torch.load('gan_test_with_model.pt')
    y_ade = d['metrics_train']['ade']
    x_ade = np.arange(len(y_ade))
    trace0 = [go.Scatter(x=x_ade,y=y_ade)]
    st.write(trace0)
    st.header('Discriminator loss')
    y_dloss = d['metrics_train']['d_loss']
    trace1 = [go.Scatter(x=x_ade, y=y_dloss)]
    st.write(trace1)
    st.header('Generator L2 loss')
    y_gloss = d['metrics_train']['g_l2_loss_abs']
    trace2 = [go.Scatter(x=x_ade,y=y_gloss)]
    st.write(trace2)
    st.header('Generator-discriminator loss')
    y_gdloss = d['G_losses']['G_discriminator_loss']
    x_gdloss = np.arange(len(y_gdloss))
    trace3 = [go.Scatter(x=x_gdloss, y=y_gdloss)]
    st.write(trace3)
    
    

