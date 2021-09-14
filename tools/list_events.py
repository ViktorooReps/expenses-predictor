from collections import defaultdict
from pprint import pprint

import pandas as pd

if __name__ == '__main__':
    print('reading csv...')
    data = pd.read_csv('cache/tinkoff_hackathon_data/avk_hackathon_data_story_logs.csv')
    print('finished reading csv! data columns:')
    print(data.columns)
    print('Unique values in category column:')
    print(data['event'].unique())

    cnts = defaultdict(int)
    for index, row in data.iterrows():
        cnts[row['event']] += 1

    print('Counts:')
    pprint(cnts)
