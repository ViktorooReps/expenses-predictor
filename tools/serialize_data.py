import datetime
import json
from collections import defaultdict
from typing import Tuple, Dict, List, Set, Optional, Any

import pandas as pd
from tqdm import tqdm

from data.datamodel import Gender, MaritalStatus, Category, Value, Impression, Date, TimeStamp, Expenses
from data.json_model import UserDataModel, TimeStampModel, StoryModel, UserDataModelList, StoryModelList

if __name__ == '__main__':
    print('Reading cache/tinkoff_hackathon_data/avk_hackathon_data_account_x_balance.csv...')
    account_x_balance = pd.read_csv('cache/tinkoff_hackathon_data/avk_hackathon_data_account_x_balance.csv')
    print('CSV was read! Columns: ', tuple(account_x_balance.columns), '\n')

    print('Reading cache/tinkoff_hackathon_data/avk_hackathon_data_party_products.csv...')
    party_products = pd.read_csv('cache/tinkoff_hackathon_data/avk_hackathon_data_party_products.csv')
    print('CSV was read! Columns: ', tuple(party_products.columns), '\n')

    print('Reading cache/tinkoff_hackathon_data/avk_hackathon_data_party_x_socdem.csv...')
    party_x_socdem = pd.read_csv('cache/tinkoff_hackathon_data/avk_hackathon_data_party_x_socdem.csv')
    print('CSV was read! Columns: ', tuple(party_x_socdem.columns), '\n')

    print('Reading cache/tinkoff_hackathon_data/avk_hackathon_data_story_logs.csv...')
    story_logs = pd.read_csv('cache/tinkoff_hackathon_data/avk_hackathon_data_story_logs.csv')
    print('CSV was read! Columns: ', tuple(story_logs.columns), '\n')

    print('Reading cache/tinkoff_hackathon_data/avk_hackathon_data_story_texts.csv...')
    story_texts = pd.read_csv('cache/tinkoff_hackathon_data/avk_hackathon_data_story_texts.csv')
    print('CSV was read! Columns: ', tuple(story_texts.columns), '\n')

    print('Reading cache/tinkoff_hackathon_data/avk_hackathon_data_test_transactions.csv...')
    test_transactions = pd.read_csv('cache/tinkoff_hackathon_data/avk_hackathon_data_test_transactions.csv')
    print('CSV was read! Columns: ', tuple(test_transactions.columns), '\n')

    print('Reading cache/tinkoff_hackathon_data/avk_hackathon_data_train_transactions.csv...')
    train_transactions = pd.read_csv('cache/tinkoff_hackathon_data/avk_hackathon_data_train_transactions.csv')
    print('CSV was read! Columns: ', tuple(train_transactions.columns), '\n')

    # users
    id2balance_chng: Dict[int, List[Tuple[datetime.date, Value]]] = defaultdict(list)
    id2product_vector: Dict[int, List[int]] = {}
    id2gender: Dict[int, Gender] = {}
    id2age: Dict[int, int] = {}
    id2marital_status: Dict[int, MaritalStatus] = {}
    id2children: Dict[int, int] = {}
    id2region: Dict[int, int] = {}

    # stories
    sid2name: Dict[int, str] = {}
    sid2text: Dict[int, str] = {}
    id2impressions: Dict[int, List[Tuple[int, datetime.date, Impression]]] = defaultdict(list)

    # train
    id2train_transactions: Dict[int, List[Tuple[Category, datetime.date, Value]]] = defaultdict(list)

    # test
    id2test_transactions: Dict[int, List[Tuple[Category, datetime.date, Value]]] = defaultdict(list)

    collected_dates: Set[datetime.date] = set()

    def get_date(str_date: str):
        if len(str_date) > 11:
            str_date = str_date.split(' ')[0]
        new_date = datetime.date.fromisoformat(str_date)
        collected_dates.add(new_date)
        return new_date

    for index, row in tqdm(account_x_balance.dropna().iterrows(), desc='Filling in id2balance_chng...', total=account_x_balance.shape[0]):
        balance_chng = float(row['balance_chng'])
        if balance_chng is not float('nan'):
            id2balance_chng[int(row['party_rk'])].append((get_date(row['cur_month']), float(row['balance_chng'])))

    for index, row in tqdm(party_products.dropna().iterrows(), desc='Filling in id2product_vector...', total=party_products.shape[0]):
        id2product_vector[int(row['party_rk'])] = [int(row['product' + str(pnum)]) for pnum in range(1, 8)]

    for index, row in tqdm(party_x_socdem.iterrows(), desc='Filling in id2gender, id2age, id2marital_status, id2children, id2region...',
                           total=party_x_socdem.shape[0]):
        party_id = int(row['party_rk'])
        id2age[party_id] = int(row['age'])
        ms = row['marital_status_desc']
        id2marital_status[party_id] = MaritalStatus(ms) if not pd.isna(ms) else MaritalStatus.UNKNOWN
        id2children[party_id] = int(row['children_cnt'])
        id2region[party_id] = int(row['region_flg'])
        gen = row['gender_cd']
        id2gender[party_id] = Gender(gen) if not pd.isna(gen) else Gender.UNKNOWN

    for index, row in tqdm(story_texts.iterrows(), desc='Filling in id2name, id2text...', total=story_texts.shape[0]):
        story_id = int(row['story_id'])
        sid2name[story_id] = str(row['name'])
        sid2text[story_id] = str(row['story_text'])

    for index, row in tqdm(story_logs.iterrows(), desc='Filling in id2impressions...', total=story_logs.shape[0]):
        pid = int(row['party_rk'])
        sid = int(row['story_id'])
        date = get_date(row['date_time'])
        imp = row['event']
        try:
            id2impressions[pid].append((sid, date, Impression(imp)))
        except ValueError:
            pass

    for index, row in tqdm(train_transactions.iterrows(), desc='Filling in id2train_transactions...', total=train_transactions.shape[0]):
        pid = int(row['party_rk'])
        date = get_date(row['transaction_dttm'])
        amount = float(row['transaction_amt_rur'])
        ct = str(row['category'])
        category = Category(ct) if not ct == 'nan' else Category.UNKNOWN
        id2train_transactions[pid].append((category, date, amount))

    for index, row in tqdm(test_transactions.iterrows(), desc='Filling in id2test_transactions...', total=test_transactions.shape[0]):
        pid = int(row['party_rk'])
        date = get_date(row['transaction_dttm'])
        amount = float(row['transaction_amt_rur'])
        ct = str(row['category'])
        category = Category(ct) if not ct == 'nan' else Category.UNKNOWN
        id2test_transactions[pid].append((category, date, amount))

    min_date = min(collected_dates)

    def diff_month(d1: datetime.date, d2: datetime.date) -> Date:
        return (d1.year - d2.year) * 12 + d1.month - d2.month

    def transform_date(full_date: datetime.date) -> Date:
        return diff_month(full_date, min_date)

    def create_timeline(user_id: int, id2trans: Dict[int, List[Tuple[Category, datetime.date, Value]]]) -> Tuple[TimeStampModel, ...]:
        timestamps: List[TimeStamp] = []
        for cat, dt, val in id2trans.get(user_id, {}):
            timestamps.append(TimeStamp(date=transform_date(dt), expenses={cat: val}))
        for st_id, dt, impression in id2impressions.get(user_id, {}):
            timestamps.append(TimeStamp(date=transform_date(dt), impressions={st_id: impression}))
        for dt, val in id2balance_chng.get(user_id, {}):
            timestamps.append(TimeStamp(date=transform_date(dt), balance_change=val))

        timestamps = sorted(timestamps, key=lambda _ts: _ts.date)

        def merge_expenses(exp1: Expenses, exp2: Expenses) -> Expenses:
            res: Expenses = defaultdict(float)
            for _cat, _val in exp1.items():
                res[_cat] += _val
            for _cat, _val in exp2.items():
                res[_cat] += _val
            return res

        def merge_time_stamps(ts1: TimeStamp, ts2: TimeStamp) -> TimeStamp:
            if ts1.date != ts2.date:
                raise ValueError
            balance_change = ts1.balance_change + ts2.balance_change
            impressions = {**ts1.impressions, **ts2.impressions}
            return TimeStamp(date=ts1.date, impressions=impressions, balance_change=balance_change,
                             expenses=merge_expenses(ts1.expenses, ts2.expenses))

        new_timestamps: List[TimeStamp] = []
        merged: Optional[TimeStamp] = None
        for ts in timestamps:
            if merged is None:
                merged = ts
            else:
                if merged.date == ts.date:
                    merged = merge_time_stamps(merged, ts)
                else:
                    new_timestamps.append(merged)
                    merged = ts

        if merged is not None:
            new_timestamps.append(merged)

        return tuple(TimeStampModel.from_time_stamp(ts) for ts in new_timestamps)

    def create_user_model(user_id: int, id2trans: Dict[int, List[Tuple[Category, datetime.date, Value]]]) -> UserDataModel:
        return UserDataModel.construct(id=user_id, gender=id2gender[user_id], age=id2age[user_id],
                                       marital_status=id2marital_status[user_id], children=id2children[user_id], region=id2region[user_id],
                                       product_vector=id2product_vector[user_id], timeline=create_timeline(user_id, id2trans))

    print('Creating JSON models...')

    train_user_models: List[UserDataModel] = [create_user_model(user_id, id2train_transactions) for user_id in id2train_transactions]
    test_user_models: List[UserDataModel] = [create_user_model(user_id, id2test_transactions) for user_id in id2test_transactions]
    story_models: List[StoryModel] = [StoryModel.construct(id=sid, name=sid2name[sid], text=sid2text[sid]) for sid in sid2name]

    print('Saving models on disc...')

    with open('data/json/users_train.json', 'w') as f:
        f.write(UserDataModelList.construct(__root__=train_user_models).json())
    with open('data/json/users_test.json', 'w') as f:
        f.write(UserDataModelList.construct(__root__=test_user_models).json())
    with open('data/json/stories.json', 'w') as f:
        f.write(StoryModelList.construct(__root__=story_models).json())
