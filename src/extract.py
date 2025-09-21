import os
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

api.dataset_download_files('blastchar/telco-customer-churn', path='data/raw', unzip=True)
