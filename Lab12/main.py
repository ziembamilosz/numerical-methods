import numpy as np
from typing import Union, Callable, List


def solve_euler(fun: Callable, t_span: np.array, y0: np.array):
    ''' 
    Funkcja umożliwiająca rozwiązanie układu równań różniczkowych z wykorzystaniem metody Eulera w przód.
    
    Parameters:
    fun: Prawa strona równania. Podana funkcja musi mieć postać fun(t, y). 
    Tutaj t jest skalarem i istnieją dwie opcje dla ndarray y: Może mieć kształt (n,); wtedy fun musi zwrócić array_like z kształtem (n,). 
    Alternatywnie może mieć kształt (n, k); wtedy fun musi zwrócić tablicę typu array_like z kształtem (n, k), tj. każda kolumna odpowiada jednej kolumnie w y. 
    t_span: wektor czasu dla którego ma zostać rozwiązane równanie
    y0: warunke początkowy równanai o wymiarze (n,)
    Results:
    (np.array): macierz o wymiarze (n,m) zawierająca w wkolumnach kolejne rozwiązania fun w czasie t_span.  

    '''
    
    if not isinstance(fun, Callable) or not isinstance(t_span, (List, np.ndarray)) or not isinstance(y0, (List, np.ndarray)):
        return None
    
    h = t_span[1] - t_span[0]
    
    if len(y0.shape) == 2:
        result = np.zeros(shape=(t_span.shape[0]+1, y0.shape[1]))
        result[0] = y0
        for i, t in enumerate(t_span):
            result[i+1] = result[i] + h*fun(t, result[i])
        return result[:-1]
    else:
        result = np.zeros(shape=(t_span.shape[0]+1))
        result[0] = y0
        for i, t in enumerate(t_span):
            result[i+1] = result[i] + h*fun(t, result[i])
        return result[:-1]



