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


def top_five_categories(arr: np.ndarray) -> np.ndarray:
    top5 = np.ravel(arr.argsort(axis=0)[-5:][::-1])
    return top5
