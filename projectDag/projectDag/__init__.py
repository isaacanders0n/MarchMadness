from dagster import (Definitions, 
                     load_assets_from_modules,
                     load_assets_from_package_module,
                     define_asset_job,
                     AssetSelection)

from . import assets
from .assets import ETL, model, EDA

all_assets = load_assets_from_modules([assets])

etl_assets = load_assets_from_package_module(ETL, 
                                            group_name = 'ETL',
                                        )

model_assets = load_assets_from_package_module(model, 
                                            group_name = 'classifier')

#still working on the eda tab
# eda_assets = load_assets_from_package_module(EDA,
#                                             group_name = 'EDA')

#cleaning = define_asset_job('cleaning', assets = AssetSelection('ETL.ncaa_cleaned', 'ETL.ncaa_rankings'))


defs = Definitions(
    assets=[*etl_assets, *model_assets],
)
