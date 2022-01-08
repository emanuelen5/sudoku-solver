import numpy as np
import hough


class HoughAccumulator:
    def __init__(self, img: np.ndarray):
        self.acc = hough.houghaccum(img)
        self.rows, self.cols = self.acc.shape
        self.rho_offset = int((self.cols - 1) / 2)

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
