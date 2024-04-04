from dagster import (Definitions, 
                     load_assets_from_modules,
                     define_asset_job,
                     AssetSelection)

from . import assets
from assets import grab_dataset

all_assets = load_assets_from_modules([assets])

background_data = define_asset_job('source_data', selection = grab_dataset)


defs = Definitions(
    assets=background_data,
)
