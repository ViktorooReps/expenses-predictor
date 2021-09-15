from collections import defaultdict
from typing import List, Dict, Optional

from pydantic import parse_file_as

from data.datamodel import Date
from data.json_model import UserDataModel

if __name__ == '__main__':
    print('Deserializing users...')
    train = parse_file_as(List[UserDataModel], 'data/json/users_train.json')
    test = parse_file_as(List[UserDataModel], 'data/json/users_test.json')

    def list_factory():
        return [None] * 2

    print('Building id2ttm...')

    id2ttm: Dict[int, List[Optional[UserDataModel]]] = defaultdict(list_factory)
    train_idx = 0
    test_idx = 1

    for udm in train:
        id2ttm[udm.id][train_idx] = udm
    for udm in test:
        id2ttm[udm.id][test_idx] = udm

    print('Collecting info...')

    def get_date(_udm: UserDataModel, date_idx: int) -> Date:
        return _udm.timeline[date_idx].date

    cnt_id_mismatch = 0
    date_mismatch = 0
    cnt_in_train_not_in_test = 0
    cnt_in_test_not_in_train = 0
    for uid, (train_udm, test_udm) in id2ttm.items():
        if train_udm is None or test_udm is None:
            cnt_id_mismatch += 1
        elif get_date(train_udm, -1) + 1 != get_date(test_udm, 0):
            date_mismatch += 1
        if train_udm is not None and test_udm is None:
            cnt_in_train_not_in_test += 1
        if train_udm is None and test_udm is not None:
            cnt_in_test_not_in_train += 1

    print(f'Mismatching ids: {cnt_id_mismatch} (in train and not in test: {cnt_in_train_not_in_test}, '
          f'in test and not in train: {cnt_in_test_not_in_train}). Mismatching dates: {date_mismatch}.')
