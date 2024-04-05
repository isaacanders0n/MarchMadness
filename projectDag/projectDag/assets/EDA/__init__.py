import pandas as pd
import sys
import os
from dagster import asset, MaterializeResult, MetadataValue, SourceAsset, AssetKey, AssetIn
from ncaa_mod import cleaning as c
# from projectDag.assets.ETL import ncaa_cleaned
import plotly.express as px
import plotly.io as pio
import base64


@asset(ins = {'ncaa_cleaned': AssetIn('ncaa_cleaned')})
def vis_dataset(ncaa_cleaned: pd.DataFrame):
    '''Preview of our data'''
    return ncaa_cleaned

@asset()
def corr(vis_dataset: pd.DataFrame):
    '''Correlation Matrix of our data'''
    #plot scatter matrix
    fig = px.scatter_matrix(vis_dataset, dimensions = ['ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D', 'TOR'])
    
    #save in memory
    bytes_img = pio.to_image(fig, format="png", width=800, height=600, scale=2.0)
    image_data = base64.b64encode(bytes_img)
    #convert to markdown for preview
    md_content = f"![img](data:image/png;base64,{image_data.decode()})"

    return MaterializeResult(metadata={"plot": MetadataValue.md(md_content)})

@asset()
def boxplot(vis_dataset: pd.DataFrame):
    '''Boxplot of our data'''
    #plot boxplot
    fig = px.box(vis_dataset, x = 'POSTSEASON', y = 'ADJOE')
    
    #save in memory
    bytes_img = pio.to_image(fig, format="png", width=800, height=600, scale=2.0)
    image_data = base64.b64encode(bytes_img)
    #convert to markdown for preview
    md_content = f"![img](data:image/png;base64,{image_data.decode()})"

    return MaterializeResult(metadata={"plot": MetadataValue.md(md_content)})