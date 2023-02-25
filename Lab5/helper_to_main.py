import numpy as np
import scipy as sp
from scipy import linalg
from  datetime import datetime
import pickle

from typing import Union, List, Tuple


def spare_matrix_Abt(m: int,n: int):
    """Funkcja tworząca zestaw składający się z macierzy A (m,n), wektora b (m,)  i pomocniczego wektora t (m,) zawierających losowe wartości
    Parameters:
    m(int): ilość wierszy macierzy A
    n(int): ilość kolumn macierzy A
    Results:
    (np.ndarray, np.ndarray): macierz o rozmiarze (m,n) i wektorem (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """

    if not isinstance(m, int) or not isinstance(n, int) or n <= 0 or m <= 0:
        return None

    t = np.linspace(0, 1, m)
    b = np.array([np.cos(4*t_value) for t_value in t])

    A = np.vander(t, N=n)
    
    for _ in range(3):
        A = np.fliplr(A)

    return A, b
