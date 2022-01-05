import numpy as np
from sudoku import SudokuBoard


test_board = SudokuBoard(np.array([
    [0, 0, 0, 0, 9, 0, 3, 0, 8],
    [2, 0, 9, 5, 0, 8, 0, 6, 1],
    [0, 1, 3, 2, 0, 0, 7, 0, 0],
    [9, 4, 0, 1, 0, 5, 0, 0, 2],
    [0, 0, 6, 3, 0, 7, 9, 0, 5],
    [3, 0, 0, 0, 0, 0, 0, 7, 0],
    [7, 0, 0, 0, 0, 0, 6, 0, 4],
    [0, 0, 4, 0, 1, 3, 0, 8, 7],
    [6, 2, 0, 0, 7, 9, 0, 0, 0]
]))

test_board_solution = SudokuBoard(np.array([
    [4, 6, 5, 7, 9, 1, 3, 2, 8],
    [2, 7, 9, 5, 3, 8, 4, 6, 1],
    [8, 1, 3, 2, 4, 6, 7, 5, 9],
    [9, 4, 7, 1, 6, 5, 8, 3, 2],
    [1, 8, 6, 3, 2, 7, 9, 4, 5],
    [3, 5, 2, 9, 8, 4, 1, 7, 6],
    [7, 3, 1, 8, 5, 2, 6, 9, 4],
    [5, 9, 4, 6, 1, 3, 2, 8, 7],
    [6, 2, 8, 4, 7, 9, 5, 1, 3]
]))


def test_available_numbers():
    assert test_board.available_numbers(0, 0) == {4, 5}


def test_solve():
    assert np.all(test_board.solve().numbers == test_board_solution.numbers)
