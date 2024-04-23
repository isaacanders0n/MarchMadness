import pandas as pd
import matplotlib.pyplot as plt
from dagster import asset, MaterializeResult, MetadataValue, SourceAsset, AssetKey, AssetIn
from ncaa_mod import cleaning as c
# from projectDag.assets.ETL import ncaa_cleaned
import plotly.express as px
import plotly.io as pio
import base64
import os

DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data'))


@asset(ins = {'ncaa_cleaned': AssetIn('ncaa_cleaned')})
def vis_dataset(ncaa_cleaned):
    '''Preview of our data'''
    return pd.read_csv(f'{DATA_FOLDER}/processed/cleaned_data.csv')

@asset()
def corr(vis_dataset: pd.DataFrame):
    '''Correlation Matrix of our data'''
    #plot scatter matrix
    fig = px.scatter_matrix(vis_dataset, dimensions = ['ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D', 'TOR'])
    
    #save in memory
    bytes_img = pio.to_image(fig, format="png", width=1280, height=720, scale=1)
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
    bytes_img = pio.to_image(fig, format="png", width=1280, height=720, scale=1)
    image_data = base64.b64encode(bytes_img)
    #convert to markdown for preview
    md_content = f"![img](data:image/png;base64,{image_data.decode()})"

    return MaterializeResult(metadata={"plot": MetadataValue.md(md_content)})

@asset()
def bubble_chart(vis_dataset: pd.DataFrame):
    df_made_postseason = vis_dataset[vis_dataset['MADE_POSTSEASON'] == 1]
    selected_columns = ['ADJOE', 'ADJDE', 'SEED']
    df_bubble = df_made_postseason[selected_columns].copy()
    max_seed = vis_dataset['SEED'].max()
    
    fig = px.scatter(df_bubble, x="ADJOE", y="ADJDE", size=(max_seed - df_bubble['SEED'] + 1) * 100, 
                     size_max=60, hover_name="SEED", color="SEED")
    fig.update_layout(
        title="Team Efficiencies and Likelihood of Higher Seed",
        xaxis_title="Offensive Efficiency",
        yaxis_title="Defensive Efficiency"
    )

    #save in memory
    bytes_img = pio.to_image(fig, format="png", width=1280, height=720, scale=1)
    image_data = base64.b64encode(bytes_img)
    #convert to markdown for preview
    md_content = f"![img](data:image/png;base64,{image_data.decode()})"
    
    return MaterializeResult(metadata={"plot": MetadataValue.md(md_content)})
