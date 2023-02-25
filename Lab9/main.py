import numpy as np
import scipy as sp
import pickle

from typing import Union, List, Tuple, Optional


def diag_dominant_matrix_A_b(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    Macierz A ma być diagonalnie zdominowana, tzn. wyrazy na przekątnej sa wieksze od pozostałych w danej kolumnie i wierszu
    Parameters:
    m int: wymiary macierzy i wektora
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: macierz diagonalnie zdominowana o rozmiarze (m,m) i wektorem (m,)
                                   Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """

    if not isinstance(m, int) or m <= 0:
        return None
    
    b = np.random.randint(0, 10, size=m)
    A = np.random.randint(0, 10, size=(m, m))
    for i in range(m):
        A[i][i] = 1000 + A[i][i]
            
    return A, b


def is_diag_dominant(A: np.ndarray) -> bool:
    """Funkcja sprawdzająca czy macierzy A (m,m) jest diagonalnie zdominowana
    Parameters:
    A np.ndarray: macierz wejściowa
    
    Returns:
    bool: sprawdzenie warunku 
          Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(A, (List, np.ndarray)) or len(A.shape) < 2 or A.shape[0]!=A.shape[1]:
        return None
    
    m = A.shape[0]
    for i in range(m):
        diag_value = A[i][i]
        sum_of_row = 0
        sum_of_col = 0
        for j in range(m):
            if j != i:
                sum_of_row += A[i][j]
                sum_of_col += A[j][i]
        if diag_value <= sum_of_row or diag_value <= sum_of_col:
            return False
    return True


def symmetric_matrix_A_b(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    Parameters:
    m int: wymiary macierzy i wektora
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: symetryczną macierz o rozmiarze (m,m) i wektorem (m,)
                                   Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(m, int) or m <= 0:
        return None

    A = np.random.randint(0, 10, size=(m, m))
    b = np.random.randint(0, 10, size=m)
    for i in range(m):
        for j in range(m):
            if i != j:
                A[j][i] = A[i][j]
            else:
                A[i][i] += 1
    return A, b


def is_symmetric(A: np.ndarray) -> bool:
    """Funkcja sprawdzająca czy macierzy A (m,m) jest symetryczna
    Parameters:
    A np.ndarray: macierz wejściowa
    
    Returns:
    bool: sprawdzenie warunku 
          Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(A, (List, np.ndarray)) or len(A.shape) < 2 or A.shape[0]!=A.shape[1]:
        return None

    m = A.shape[0]
    for i in range(m):
        for j in range(m):
            if i != j:
                if A[i][j] != A[j][i]:
                    return False
    return True


def solve_jacobi(A: np.ndarray, b: np.ndarray, x_init: np.ndarray,
                 epsilon: Optional[float] = 1e-8, maxiter: Optional[int] = 100) -> Tuple[np.ndarray, int]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych
    Parameters:
    A np.ndarray: macierz współczynników
    b np.ndarray: wektor wartości prawej strony układu
    x_init np.ndarray: rozwiązanie początkowe
    epsilon Optional[float]: zadana dokładność
    maxiter Optional[int]: ograniczenie iteracji
    
    Returns:
    np.ndarray: przybliżone rozwiązanie (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    int: iteracja
    """

    if not isinstance(A, (List, np.ndarray)) or not isinstance(b, (List, np.ndarray)) or not isinstance(x_init, (List, np.ndarray)) \
        or not isinstance(maxiter, int) or not isinstance(epsilon, float) or epsilon < 0 \
        or maxiter < 0 or A.shape[0]!=b.shape[0] or b.shape[0]!=x_init.shape[0]:
        return None

    D = np.diag(np.diag(A))
    LU = A - D
    x = x_init
    D_inv = np.diag(1 / np.diag(D))
    resid=[]
    iter = 0
    for i in range(maxiter):
        iter += 1
        x_new = np.dot(D_inv, b - np.dot(LU, x))
        r_norm = np.linalg.norm(x_new - x)
        resid.append(r_norm)
        if  r_norm < epsilon:
            return np.array(x_new), iter
        x = x_new
    return np.array(x), iter