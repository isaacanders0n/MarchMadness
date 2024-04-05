
#still working on this
# import pandas as pd
# import sys
# import os
# from dagster import asset, MaterializeResult, MetadataValue
# from ncaa_mod import cleaning as c
# from projectDag.assets.ETL import ncaa_cleaned
# import plotly.express as px
# import plotly.io as pio
# import base64


# @asset()
# def corr():
#     '''Correlation Matrix of our data'''
#     df = ncaa_cleaned()
#     #plot scatter matrix
#     fig = px.scatter_matrix(df, dimensions = ['ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D', 'TOR'])
    
#     #save in memory
#     bytes_img = pio.to_image(fig, format="png", width=800, height=600, scale=2.0)
#     image_data = base64.b64encode(bytes_img)
#     #convert to markdown for preview
#     md_content = f"![img](data:image/png;base64,{image_data.decode()})"

#     return MaterializeResult(metadata={"plot": MetadataValue.md(md_content)})
