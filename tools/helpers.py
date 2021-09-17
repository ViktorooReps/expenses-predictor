from typing import List, Tuple

import numpy as np

from data.datamodel import Expenses, Category


def expenses_to_np_array(expenses: Expenses) -> np.ndarray:
    total_expenses = len(Category)
    expenses_by_category: List[float] = [0] * total_expenses

    for category, expense in expenses.items():
        category: Category
        expenses_by_category[category.id] = expense

    return np.ndarray(shape=(total_expenses, 1), buffer=np.array(expenses_by_category, dtype=float))


def top_five_categories(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ind_val = np.array(sorted([[i, arr[i]] for i in range(5)], reverse=True))

    max_values = ind_val.transpose()[1]
    max_indexes = ind_val.transpose()[0]

    for i in range(5, arr.size):
        if arr[i] > max_values[0]:
            max_values = np.concatenate(([arr[i]], max_values[:4]))
            max_indexes = np.concatenate(([i], max_indexes[:4]))
        elif arr[i] > max_values[1]:
            max_values = np.concatenate(([max_values[0]], [arr[i]], max_values[2:4]))
            max_indexes = np.concatenate(([max_indexes[0]], [i], max_indexes[2:4]))
        elif arr[i] > max_values[2]:
            max_values[4] = max_values[3]
            max_values[3] = max_values[2]
            max_values[2] = arr[i]

            max_indexes[4] = max_indexes[3]
            max_indexes[3] = max_indexes[2]
            max_indexes[2] = i
        elif arr[i] > max_values[3]:
            max_values[4] = max_values[3]
            max_values[3] = arr[i]

            max_indexes[4] = max_indexes[3]
            max_indexes[3] = i
        elif arr[i] > max_values[4]:
            max_values[4] = arr[i]

            max_indexes[4] = i

    return max_indexes, max_values
