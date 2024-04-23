import pandas as pd
import os
from ncaa_mod import cleaning as c
from dagster import MetadataValue, asset, MaterializeResult, AssetIn

DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data'))



@asset(ins = {'ncaa_rankings': AssetIn('ncaa_rankings')})
def ncaa_cleaned(ncaa_rankings):
    '''Cleaned data'''
    df = c.clean_data(ncaa_rankings)
    df.to_csv(f'{DATA_FOLDER}/processed/cleaned_data.csv', index=False)

    return MaterializeResult(
        metadata={
            "num_records": len(df),  # Metsadata can be any key-value pair
            "preview": MetadataValue.md(df.head().to_markdown())})
            # The `MetadataValue` class has useful static methods to build Metadata


@asset(deps = [ncaa_cleaned])
def parameter_tuning():
    '''Parameter tuning for a more complex ML model'''
    df = pd.read_csv(f'{DATA_FOLDER}/processed/cleaned_data.csv')
    df.drop('EFG_D', axis = 1, inplace = True)
    return df.dropna(subset= ["ADJOE","ADJDE","BARTHAG","EFG_O"
                            ,"TOR","TORD","ORB","DRB","FTR","FTRD","2P_O","2P_D","3P_O","3P_D", "ADJ_T","WAB"]) 



@asset(ins = {'bubbleClassification': AssetIn('bubbleClassification')})
def predictionValidation(bubbleClassification):
    '''Evaluating the model accuracy'''

    predictions = pd.read_csv(f'{DATA_FOLDER}/output/predictions.csv')
    validation = pd.read_csv(f'{DATA_FOLDER}/validationSet.csv')

    # Merge the predictions with the validation set
    validation = pd.merge(validation, predictions, on = 'TEAM')

    # Calculate the accuracy of the model
    accuracy = (validation['Prediction'] == validation['MadePostseason']).mean()
    print(type(accuracy))
    accuracy = round(accuracy, 2).astype(str)

    return MaterializeResult(metadata={"Model Accuracy: ": accuracy})