import numpy as np
import pickle

from typing import Union, List, Tuple

def random_matrix_Ab(m:int):
    """Funkcja tworząca zestaw składający się z macierzy A (m,m) i wektora b (m,)  zawierających losowe wartości
    Parameters:
    m(int): rozmiar macierzy
    Results:
    (np.ndarray, np.ndarray): macierz o rozmiarze (m,m) i wektorem (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(m, int) or m <= 0:
      return None
    
    else:

      b = np.zeros(m)
      A = np.zeros((m, m))
      for i in range(m):
        A[i] = np.random.randint(10, size=m)
        A[i][i] += 1
      
      b = np.array(np.random.randint(10, size=m))
      return A, b

      
def residual_norm(A:np.ndarray,x:np.ndarray, b:np.ndarray):
    """Funkcja obliczająca normę residuum dla równania postaci:
    Ax = b

      Parameters:
      A: macierz A (m,m) zawierająca współczynniki równania 
      x: wektor x (m.) zawierający rozwiązania równania 
      b: wektor b (m,) zawierający współczynniki po prawej stronie równania

      Results:
      (float)- wartość normy residuom dla podanych parametrów"""
    
    if A.shape[1] != x.shape[0] or x.shape[0] != b.shape[0]:
      return None
    
    else:
      result = A@x-b
      norm_value = np.linalg.norm(result)
      return norm_value


def log_sing_value(n:int, min_order:Union[int,float], max_order:Union[int,float]):
    """Funkcja generująca wektor wartości singularnych rozłożonych w skali logarytmiczne
    
        Parameters:
         n(np.ndarray): rozmiar wektora wartości singularnych (n,), gdzie n>0
         min_order(int,float): rząd najmniejszej wartości w wektorze wartości singularnych
         max_order(int,float): rząd największej wartości w wektorze wartości singularnych
         Results:
         np.ndarray - wektor nierosnących wartości logarytmicznych o wymiarze (n,) zawierający wartości logarytmiczne na zadanym przedziale
         """
    
    if not isinstance(n, int) or n <= 0 or not isinstance(min_order, (int, float)) or not isinstance(max_order, (int, float)) or min_order > max_order:
      return None
    
    singular_values = np.logspace(min_order, max_order, num=n)
    return singular_values
    
def order_sing_value(n:int, order:Union[int,float] = 2, site:str = 'gre'):
    """Funkcja generująca wektor losowych wartości singularnych (n,) będących wartościami zmiennoprzecinkowymi losowanymi przy użyciu funkcji np.random.rand(n)*10. 
        A następnie ustawiająca wartość minimalną (site = 'low') albo maksymalną (site = 'gre') na wartość o  10**order razy mniejszą/większą.
    
        Parameters:
        n(np.ndarray): rozmiar wektora wartości singularnych (n,), gdzie n>0
        order(int,float): rząd przeskalowania wartości skrajnej
        site(str): zmienna wskazująca stronnę zmiany:
            - site = 'low' -> sing_value[-1] * 10**order
            - site = 'gre' -> sing_value[0] * 10**order
        
        Results:
        np.ndarray - wektor wartości singularnych o wymiarze (n,) zawierający wartości logarytmiczne na zadanym przedziale
        """
    if not isinstance(n, int) or not isinstance(order, (int, float)) or n <= 0 or not isinstance(site, str) or not (site == 'gre' or site == 'low'):
      return None
    
    random_singular_values = np.random.rand(n)*10
    sorted_random_singular_values = np.flip(np.sort(random_singular_values))

    if site == 'gre':
      sorted_random_singular_values[0] = (sorted_random_singular_values[0])*(10**order)
    else:
      sorted_random_singular_values[-1] = (sorted_random_singular_values[-1])*(10**order)
    
    return sorted_random_singular_values


def create_matrix_from_A(A:np.ndarray, sing_value:np.ndarray):
    """Funkcja generująca rozkład SVD dla macierzy A i zwracająca otworzenie macierzy A z wykorzystaniem zdefiniowanego wektora warości singularnych

            Parameters:
            A(np.ndarray): rozmiarz macierzy A (m,m)
            sing_value(np.ndarray): wektor wartości singularnych (m,)


            Results:
            np.ndarray: macierz (m,m) utworzoną na podstawie rozkładu SVD zadanej macierzy A z podmienionym wektorem wartości singularnych na wektor sing_valu """
    
    if not isinstance(A, np.ndarray) or not isinstance(sing_value, np.ndarray) or A.shape[0] != A.shape[1] or A.shape[0] != sing_value.shape[0]:
      return None
    
    U, _, V = np.linalg.svd(A)
    S = np.diag(sing_value)
    A2 = np.dot(U@S, V)
    return A2