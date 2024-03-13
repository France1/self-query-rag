from typing import Any, Dict

import pandas as pd
import numpy as np
from self_query_summarization.utils.utils import build_path, load_config_yaml

class DataLoader:
    # from https://www.kaggle.com/code/tatianasnwrt/wikipedia-movie-plots-eda?scriptVersionId=31690832&cellId=15
    ethnicity_to_country = {
        "American":"United States", "Australian":"Australia", "Bangladeshi":"Bangladesh", 
        "British":"United Kingdom", "Canadian":"Canada", "Chinese":"China", 
        "Egyptian":"Egypt", "Hong Kong":"Hong Kong", "Fillipino":"Phillipines", 
        "Assamese":"India", "Bengali":"India", "Bollywood":"India", "Kannada":"India", 
        "Malayalam":"India", "Marathi":"India", "Punjabi":"India", "Tamil":"India", 
        "Telugu":"India", "Japanese":"Japan", "Malaysian":"Malaysia", "Maldivian":"Maldives", 
        "Russian":"Russia", "South_Korean":"South Korea","Turkish":"Turkey"
    }
    
    column_renaming = {
        "Release Year": "year", "Title": "title", "Director": "director", "Cast": "cast", 
        "Genre": "genre", "Wiki Page": "wiki_page", "Plot": "description", "Country": "country"
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.load_data()
    
    def load_data(self) -> pd.DataFrame: 
        # load data
        dir_name, file_name = self.config['data']['dir'], self.config['data']['file']
        self.data = pd.read_csv(build_path(dir_name, file_name))
        self.n_items = len(self.data)
    
    def transform(self) -> pd.DataFrame: 
        # TODO: further cleanup columns - for example Cast have also Directors in it sometimes
        # standardize missing data
        self.data = self.data.replace([np.NaN, 'Unknown'], 'unknown')
        # remap Origin column into Country column for simplicity
        self.data["Country"] = self.data['Origin/Ethnicity'].map(self.ethnicity_to_country)
        self.data = self.data.drop(columns=['Origin/Ethnicity'])
        self.data = self.data.rename(columns=self.column_renaming)
        # add document id column
        self.data['id'] = [i for i in range(self.n_items)]

        
        return self.data