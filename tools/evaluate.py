from argparse import ArgumentParser
from os import PathLike
from typing import Optional, Tuple

import numpy as np

from data.datamodel import Dataset, Category
from models.abstract import AbstractPredictor
from models.predictors import name2predictor
from models.registered import PredictorName
from helpers import expenses_to_np_array, top_five_categories


def evaluate(model: AbstractPredictor, data_path: PathLike) -> Tuple[np.array, np.array, np.array, np.array, float]:
    dataset: Dataset = Dataset.load(data_path)
    users, expenses = dataset.data

    predict_expenses = model.predict_users(users)

    predict_array = np.array([expenses_to_np_array(expense) for expense in predict_expenses])
    test_array = np.array([expenses_to_np_array(expense) for expense in expenses])

    mae_by_category = np.mean(abs(test_array - predict_array), axis=0)

    rmsle_by_category = np.sqrt(
        np.sum((np.log(predict_array + 1) - np.log(test_array + 1)) ** 2, axis=0) / len(predict_array[0]))

    rmsle_for_all = np.sqrt(np.sum((np.log(predict_array + 1) - np.log(test_array + 1)) ** 2) / predict_array.size)

    percent_of_rank_guesses = np.array([0] * 5)
    percent_of_top_guesses = np.array([0] * len(Category))
    count_in_top = np.array([0] * len(Category))

    for user_number in range(predict_array.shape[0]):
        predict_top, predict_top_val = top_five_categories(predict_array[user_number])
        test_top, test_top_val = top_five_categories(test_array[user_number])
        percent_of_top_guesses = percent_of_rank_guesses + (predict_top == test_top) * 1

        for ind in predict_top:
            if ind in test_top:
                percent_of_top_guesses[ind] += 1

        for ind in test_top:
            count_in_top[ind] += 1

    percent_of_top_guesses = percent_of_top_guesses / count_in_top
    percent_of_rank_guesses = (percent_of_rank_guesses / predict_array.shape[0]) * 100

    return mae_by_category, percent_of_rank_guesses, percent_of_top_guesses, rmsle_by_category, rmsle_for_all


if __name__ == '__main__':
    predictor_names = [pn.value for pn in PredictorName]

    parser = ArgumentParser()
    parser.add_argument('-predictor', choices=predictor_names, type=str, default=PredictorName.STUB.value)
    parser.add_argument('-datapath', type=str)

    args = parser.parse_args()

    predictor = name2predictor[PredictorName(args.predictor)]()
    mae, per_of_rank_guesses, per_of_top_guesses, rmsle_by_category, rmsle_for_all = evaluate(predictor, args.datapath)

    print("__________________________________________________________")
    print(f"Метрики, собранные по прогнозу модели {args.predictor}:\n")
    print("__________________________________________________________")

    for index in range(mae.size):
        print(f"Категория {Category.from_id(index)}:\n")
        print(f"MAE по категории: {mae[index]}\n")
        print(f"RMSLE по категории: {rmsle_by_category[index]}\n")
        print(f"Процент верных прогнозов попадания в Топ-5: {per_of_top_guesses[index]}\n\n\n")

    print(f"MAE по всем категориям: {mae.mean()}\n\n\n")
    print(f"RMSLE по всем категориям: {rmsle_for_all}\n\n\n")

    for index in range(per_of_rank_guesses.size):
        print(f"Процент удачных прогнозов категории Топ-{index}: {per_of_rank_guesses[index]}\n\n\n")
    print("__________________________________________________________\n")

