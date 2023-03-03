'''
    PDU function needs calibration befor usage.
    PDUFunctionCreator class encapsulates calibration and usage of PDU function
'''


from copy import deepcopy
from LPIQE.src.LPIQE_reconstruction_executor import  LpiqeReconstructionExecutor
from LPIQE.src.LPIQE_representation import  LpiqeRepresentation
from scipy.interpolate import interp1d
from qiskit import transpile, execute

import numpy as np


class PDUFunctionCreator:

    def __init__(self, print_info: bool, shape: tuple, lpiqe_executor: LpiqeReconstructionExecutor, shots):
        self.__x = 0
        self.__y = 0
        self.__PDU_fun = []
        self.__print_info = print_info
        self.__lpiqe_executor = lpiqe_executor
        self.__image_shape = shape
        self.__transpiled_circuits = []
        self.__shots = shots

    def create_empty_function(self):
        '''
        Method for creating empty PDU function

        :param shape: shape of images which will be used
        '''
        self.__x, self.__y = self.__image_shape
        dict_cell = {}

        for i in range(self.__x):
            row=[]
            for j in range(self.__y):
                row.append(deepcopy(dict_cell))
            self.__PDU_fun.append(row)


    def calibrate_function(self, granularity):
        '''
        Method for calibrating PDU function

        :param granularity: granularity of calibration - the higher the better the function will be
        '''
        if self.__print_info:
            print('Calibrating PDU function')
        step = -255/(granularity-1)
        self.__grey_values = np.arange(255, step, step).tolist()
        self.__backend = self.__lpiqe_executor.backend
        for grey_value in self.__grey_values:
            img_grey=np.full(self.__image_shape, grey_value, np.uint8)/255
            self.__lpiqe_executor.create_circuit(img_grey)
            tc = transpile(self.__lpiqe_executor.circuit, self.__backend)
            self.__transpiled_circuits.append(tc)
        
        job = execute(self.__transpiled_circuits, self.__backend, shots=self.__shots)
        result = job.result()
        counts = result.get_counts()

        for i in range(len(self.__grey_values)):
            count = counts[i]
            grey_value = self.__grey_values[i]
            self.calibrate(count, grey_value)
            

        if self.__print_info:
            print('PDU function calibrated')


    def calibrate(self, counts, grey_value):
        repr = self.__lpiqe_executor.representation
        repr.results = [counts, self.__shots]
        repr.reconstruct(False)
        im_recon = repr.reconstructed_image[0]
        key = grey_value/255

        for x in range(self.__x):
            for y in range(self.__y):
                value = key - im_recon[x][y]
                self.__PDU_fun[x][y][key] = value

    def create_interpolation_functions(self):
        '''
        Method for interpolating PDU functions for every pixel
        '''
        self.__PDU_interpolated_functions = []

        for i in range(self.__x):
            row=[]
            for j in range(self.__y):
                dict = self.__PDU_fun[i][j]
                x = [key for key in dict]
                y = [value for value in dict.values()]
                f = interp1d(x, y, kind='cubic')
                row.append(f)
            self.__PDU_interpolated_functions.append(row)


    def get_interpolated_value_for_point(self, x_index, y_index, value):
        '''
        Method for getting interpolated value for specific pixel

        :param x_index: x index of pixel
        :param x_index: y index of pixel
        :param value: gray value for which function should be interpolated

        :returns: interpolated value
        '''
        if value in self.__PDU_fun[x_index][y_index]:
            return self.__PDU_fun[x_index][y_index][value]

        f = self.__PDU_interpolated_functions[x_index][y_index]

        return f(value)


    def apply(self, result_image: np.ndarray, original_image: np.ndarray):
        '''
        Method for applying PDU function to image

        :param result_image: image after encoding and reconstruction on the quantum computer
        :param original_image: original image encoded
        '''
        self.create_interpolation_functions()
        self.__corrected_image = np.full((self.__x, self.__y), 0.0)

        for x in range(self.__x):
            for y in range(self.__y):
                correction_value = self.get_interpolated_value_for_point(x, y, original_image[x][y])
                correct_value = result_image[x][y]+correction_value
                if correct_value < 0:
                    correct_value = 0
                elif correct_value > 1:
                    correct_value = 1
                self.__corrected_image[x][y] = correct_value

    @property
    def pdu_funtion(self):
        return self.__PDU_fun

    @property
    def corrected_image(self):
        return self.__corrected_image