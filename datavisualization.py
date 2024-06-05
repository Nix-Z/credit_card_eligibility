import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from data_preprocess import preprocess_data

def visualize_data():
    data, categorical_features, numerical_features = preprocess_data()

    # Create correlation plot for numerical featues
    corr_data_num = data[numerical_features].corr()
    print(corr_data_num)
    fig_1 = px.imshow(corr_data_num, labels=dict(color="Correlation"), x=corr_data_num.columns, y=corr_data_num.index, text_auto=True)
    #fig_1.show()
    fig_1.write_image('fig_1.jpg')

    # Create histogram for each categorical feature
    for categorical_feature in categorical_features:
        fig = px.histogram(data, x=categorical_feature)
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        #fig.show()
        fig.write_image(f'fig_{categorical_feature}.jpg')

    return data, categorical_features, numerical_features

# visualize_data()
