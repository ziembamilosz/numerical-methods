##
import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt

from typing import Union, List, Tuple

def chebyshev_nodes(n:int=10)-> np.ndarray:
    """Funkcja tworząca wektor zawierający węzły czybyszewa w postaci wektora (n+1,)
    
    Parameters:
    n(int): numer ostaniego węzła Czebyszewa. Wartość musi być większa od 0.
     
    Results:
    np.ndarray: wektor węzłów Czybyszewa o rozmiarze (n+1,). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(n, int) or n <= 0:
        return None
    
    chebyshev = []

    for k in range(0, n+1):
        chebyshev.append(np.cos(k*np.pi/n))
    
    return np.array(chebyshev)
    

def bar_czeb_weights(n:int=10)-> np.ndarray:
    """Funkcja tworząca wektor wag dla węzłów czybyszewa w postaci (n+1,)
    
    Parameters:
    n(int): numer ostaniej wagi dla węzłów Czebyszewa. Wartość musi być większa od 0.
     
    Results:
    np.ndarray: wektor wag dla węzłów Czybyszewa o rozmiarze (n+1,). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(n, int):
        return None
    
    if n <= 0:
        return None

    def delta_j(index):
        if index == n  or index == 0:
            return 0.5
        else: 
            return 1

    weights = []
    for i in range(n+1):
        weights.append(((-1)**i)*delta_j(i))
    
    return np.array(weights)
    
def  barycentric_inte(xi:np.ndarray,yi:np.ndarray,wi:np.ndarray,x:np.ndarray)-> np.ndarray:
    """Funkcja przprowadza interpolację metodą barycentryczną dla zadanych węzłów xi
        i wartości funkcji interpolowanej yi używając wag wi. Zwraca wyliczone wartości
        funkcji interpolującej dla argumentów x w postaci wektora (n,) gdzie n to dłógość
        wektora n. 
    
    Parameters:
    xi(np.ndarray): węzły interpolacji w postaci wektora (m,), gdzie m > 0
    yi(np.ndarray): wartości funkcji interpolowanej w węzłach w postaci wektora (m,), gdzie m>0
    wi(np.ndarray): wagi interpolacji w postaci wektora (m,), gdzie m>0
    x(np.ndarray): argumenty dla funkcji interpolującej (n,), gdzie n>0 
     
    Results:
    np.ndarray: wektor wartości funkcji interpolującej o rozmiarze (n,). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(xi, np.ndarray) or not isinstance(yi, np.ndarray) \
                or not isinstance(wi, np.ndarray) or not isinstance(x, np.ndarray) \
                or xi.shape[0] != yi.shape[0] or xi.shape[0] != wi.shape[0]:
        return None

    y = []
    
    for item in x:
        numerator = 0
        denominator = 0
        for i in range(len(xi)):
            if xi[i] != item:
                expr = (wi[i])/(item - xi[i])
                numerator += yi[i]*expr
                denominator += expr
        y.append(numerator/denominator)

    return np.array(y)

def L_inf(xr:Union[int, float, List, np.ndarray],x:Union[int, float, List, np.ndarray])-> float:
    """Obliczenie normy  L nieskończonośćg. 
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach biblioteki numpy.
    
    Parameters:
    xr (Union[int, float, List, np.ndarray]): wartość dokładna w postaci wektora (n,)
    x (Union[int, float, List, np.ndarray]): wartość przybliżona w postaci wektora (n,1)
    
    Returns:
    float: wartość normy L nieskończoność,
                                    NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(xr, (int, float)) and isinstance(x,(int,float)):
        return abs(xr - x)
    elif isinstance(xr, np.ndarray) and isinstance(x,np.ndarray) and xr.shape == x.shape:
       return max(abs(xr - x))
    elif isinstance(xr , List) and isinstance(x , List):
        return abs(max(xr) - max(x))
    else:
        return np.NaN