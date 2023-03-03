

from PDU.src.PDU_function_creator import PDUFunctionCreator
import numpy as np

class PDUStatisticsHelper:

    def __init__(self):
        self.__mse = 0
        self.__std_dev = 0

    def calculate_mse(self, original_image: np.ndarray, reconstructed_image: np.ndarray):
        x, y = original_image.shape
        difference_image = original_image - reconstructed_image
        squared_difference_image = difference_image**2
        self.__mse = squared_difference_image.mean()

    def calculate_std_dev(self, original_image: np.ndarray, reconstructed_image: np.ndarray):
        difference_image = original_image - reconstructed_image
        self.__std_dev = np.std(difference_image)

    def calculate(self, original_image: np.ndarray, reconstructed_image: np.ndarray):
        self.calculate_mse(original_image, reconstructed_image)
        self.calculate_std_dev(original_image, reconstructed_image)

    @property
    def mse(self):
        return self.__mse

    @property
    def std_dev(self):
        return self.__std_dev