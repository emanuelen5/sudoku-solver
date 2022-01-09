import copy
import numpy as np
import hough


class HoughAccumulator:
    def __init__(self, img: np.ndarray):
        self.acc = hough.houghaccum(img)
        self.rho_per_col = 1
        self.theta_per_col = 1
        self.rows, self.cols = self.acc.shape
        self.rho_offset = int((self.cols - 1) / 2)
        self.rows *= 2
        mirror = self.acc.copy()
        mirror = np.hstack((np.fliplr(mirror[:, -self.rho_offset:]), mirror[:, self.rho_offset:self.rho_offset+1], np.fliplr(mirror[:, :self.rho_offset])))
        self.acc = np.vstack((self.acc, mirror))

    def col_to_rho(self, col: int) -> float:
        return ((col % self.cols) - self.rho_offset) * self.rho_per_col

    def row_to_theta(self, row: int) -> float:
        return (row * self.theta_per_col) % 360

    def theta_to_row(self, theta: float) -> int:
        return int(round(theta)) % self.rows

    def rho_to_col(self, rho: float) -> int:
        return int(round(rho + self.rho_offset)) % self.cols

    def __getitem__(self, name) -> np.ndarray:
        theta, rho = name
        if isinstance(rho, slice):
            cols = slice(self.rho_to_col(rho.start), self.rho_to_col(rho.stop), rho.step)
        else:
            cols = self.rho_to_col(rho)

        if isinstance(theta, slice):
            rows = slice(self.theta_to_row(rho.start), self.theta_to_row(rho.stop), rho.step)
        else:
            rows = self.theta_to_row(theta)
        return self.acc[rows, cols]
    
    def copy(self) -> "HoughAccumulator":
        return copy.deepcopy(self)
