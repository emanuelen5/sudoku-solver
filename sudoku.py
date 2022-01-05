from dataclasses import dataclass
from typing import Union
import numpy as np


all_numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9}


@dataclass
class SudokuBoard:
    numbers: np.ndarray

    def __post_init__(self):
        assert self.numbers.shape == (9, 9)

    def check_possible(self, i, j, num: int) -> bool:
        row_numbers = self.numbers[i, :]
        if num not in all_numbers.difference(set(row_numbers)):
            return False
        col_numbers = self.numbers[:, j]
        if num not in all_numbers.difference(set(col_numbers)):
            return False
        block_row = (i // 3) * 3
        block_col = (j // 3) * 3
        block_numbers = self.numbers[block_row:block_row+3, block_col:block_col+3].flatten()
        if num not in all_numbers.difference(set(block_numbers)):
            return False
        return True

    def available_numbers(self, i, j) -> set[int]:
        row_numbers = all_numbers.difference(set(self.numbers[i, :]))
        col_numbers = all_numbers.difference(set(self.numbers[:, j]))
        block_row = (i // 3) * 3
        block_col = (j // 3) * 3
        block_numbers = all_numbers.difference(set(self.numbers[block_row:block_row+3, block_col:block_col+3].flatten()))
        return row_numbers.intersection(col_numbers).intersection(block_numbers)

    def solve(self) -> Union["SudokuBoard", None]:
        rows, cols = self.numbers.shape
        for i in range(rows):
            for j in range(cols):
                if self.numbers[i, j] != 0:
                    continue
                available = self.available_numbers(i, j)
                if len(available) == 0:
                    return
                for num in available:
                    solution = self.set(i, j, num).solve()
                    if solution:
                        return solution
                else:
                    return
        return self

    def set(self, i, j, num: int) -> "SudokuBoard":
        assert self.check_possible(i, j, num)
        numbers = self.numbers.copy()
        numbers[i, j] = num
        return SudokuBoard(numbers)

    def copy(self) -> "SudokuBoard":
        return SudokuBoard(self.numbers.copy())

    def __repr__(self):
        rows, cols = self.numbers.shape
        print_width = cols * 3 + 4
        s = ""
        for i in range(rows):
            for j in range(cols):
                num = self.numbers[i, j]
                if (i % 3 == 0) and j == 0:
                    s += "=" * print_width + "\n"
                if j % 3 == 0:
                    s += "|"
                if num:
                    s += f" {num} "
                else:
                    s += "   "
                if j == 8:
                    s += "|\n"
                if i == 8 and j == 8:
                    s += "=" * print_width + "\n"
        return s
