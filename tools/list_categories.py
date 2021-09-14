import pandas as pd

if __name__ == '__main__':
    print('reading csv...')
    data = pd.read_csv('cache/tinkoff_hackathon_data/avk_hackathon_data_train_transactions.csv')
    print('finished reading csv! data columns:')
    print(data.columns)
    print('Unique values in category column:')
    print(list(data['category'].unique()))

    print('reading csv...')
    data = pd.read_csv('cache/tinkoff_hackathon_data/avk_hackathon_data_test_transactions.csv')
    print('finished reading csv! data columns:')
    print(data.columns)
    print('Unique values in category column:')
    print(list(data['category'].unique()))
