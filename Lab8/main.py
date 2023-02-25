import numpy as np
import scipy as sp
from scipy import linalg
from  datetime import datetime
import pickle

from typing import Union, List, Tuple

'''
Do celów testowych dla elementów losowych uzywaj seed = 24122022
'''

def random_matrix_by_egval(egval_vec: np.ndarray):
    """Funkcja z pierwszego zadania domowego
    Parameters:
    egval_vec : wetkor wartości własnych
    Results:
    np.ndarray: losowa macierza o zadanych wartościach własnych 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(egval_vec, (np.ndarray, List)) or isinstance(egval_vec[0], str):
        return None

    np.random.seed(24122022)

    J = np.diag(egval_vec)
    size = len(egval_vec)
    P = np.random.rand(size, size)
    P_inversed = np.linalg.inv(P)
    A = P@J@P_inversed

    return A


def frob_a(coef_vec: np.ndarray):
    """Funkcja z drugiego zadania domowego
    Parameters:
    coef_vec : wetkor wartości wspołczynników
    Results:
    np.ndarray: macierza Frobeniusa o zadanych wartościach współczynników wielomianu 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(coef_vec, np.ndarray):
        return None

    size = coef_vec.shape[0]
    F = np.eye(size, k=1)
    F[-1] = -coef_vec[::-1]
    return F

    
def polly_from_egval(egval_vec: np.ndarray):
    """Funkcja z laboratorium 8
    Parameters:
    egval_vec: wetkor wartości własnych
    Results:
    np.ndarray: wektor współczynników wielomianu charakterystycznego
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(egval_vec, (np.ndarray, List)):
        return None
    
    return np.poly(egval_vec)