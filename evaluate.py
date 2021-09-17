from argparse import ArgumentParser
from os import PathLike
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

from data.datamodel import Dataset, Category
from models.abstract import AbstractPredictor
from models.predictors import name2predictor
from models.registered import PredictorName
from tools.helpers import expenses_to_np_array, top_five_categories


def evaluate(model: AbstractPredictor, dataset: Dataset) -> Tuple[np.array, np.array, np.array, np.array, float]:
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

    for user_number in tqdm(range(predict_array.shape[0]), total=predict_array.shape[0], leave=False):
        predict_top = top_five_categories(predict_array[user_number])
        test_top = top_five_categories(test_array[user_number])
        percent_of_rank_guesses += (predict_top == test_top) * 1

        correct_guesses = list(set(predict_top).intersection(set(test_top)))
        percent_of_top_guesses[correct_guesses] += 1

        count_in_top[test_top] += 1

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
    dataset = Dataset.load(args.datapath)
    mae, per_of_rank_guesses, per_of_top_guesses, rmsle_by_category, rmsle_for_all = evaluate(predictor, dataset)

    print("__________________________________________________________")
    print(f"Метрики, собранные по прогнозу модели {args.predictor}:")
    print("__________________________________________________________\n")

    for index in range(mae.size):
        print(f"Категория {Category.from_id(index)}:")
        print(f"MAE по категории: {mae[index]}")
        print(f"RMSLE по категории: {rmsle_by_category[index]}")
        print(f"Процент верных прогнозов попадания в Топ-5: {per_of_top_guesses[index]}\n")

    print(f"MAE по всем категориям: {np.mean(mae)}")
    print(f"RMSLE по всем категориям: {np.mean(rmsle_by_category)}\n")

    for index in range(per_of_rank_guesses.size):
        print(f"Процент удачных прогнозов категории Топ-{index}: {per_of_rank_guesses[index]}")
    print("__________________________________________________________\n")
