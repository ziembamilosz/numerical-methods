import numpy as np
import scipy
import pickle
import typing
import math
import types
import pickle 
from inspect import isfunction


from typing import Union, List, Tuple

def fun(x):
    return np.exp(-2*x)+x**2-1

def dfun(x):
    return -2*np.exp(-2*x) + 2*x

def ddfun(x):
    return 4*np.exp(-2*x) + 2


def bisection(a: Union[int,float], b: Union[int,float], f: typing.Callable[[float], float], epsilon: float, iteration: int) -> Tuple[float, int]:
    '''funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą bisekcji.

    Parametry:
    a - początek przedziału
    b - koniec przedziału
    f - funkcja dla której jest poszukiwane rozwiązanie
    epsilon - tolerancja zera maszynowego (warunek stopu)
    iteration - ilość iteracji

    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or not isinstance(epsilon, (int, float)) or not isinstance(iteration, int):
        return None
    
    if f(a)*f(b) > 0 or a >= b:
        return None

    c = (a+b)/2
    nr_of_iterations = 0
    
    while(nr_of_iterations <= iteration):
        
        c = (a+b)/2
        
        if(np.abs(f(c)) < epsilon):
            break
        
        nr_of_iterations += 1

        if(f(a)*f(c)<0):
            b = c
        else:
            a = c

    return c, nr_of_iterations
    

def secant(a: Union[int,float], b: Union[int,float], f: typing.Callable[[float], float], epsilon: float, iteration: int) -> Tuple[float, int]:
    '''funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą siecznych.

    Parametry:
    a - początek przedziału
    b - koniec przedziału
    f - funkcja dla której jest poszukiwane rozwiązanie
    epsilon - tolerancja zera maszynowego (warunek stopu)
    iteration - ilość iteracji

    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or not isinstance(epsilon, (int, float)) or not isinstance(iteration, int):
        return None
    
    if f(a)*f(b) > 0 or a >= b:
        return None

    nr_of_iterations = -1
    
    while(nr_of_iterations < iteration):
        
        if(f(b) != f(a)):
            c = (f(b)*a-f(a)*b)/(f(b)-f(a))
        
        if(np.abs(f(c)) < epsilon):
            return c, nr_of_iterations + 1
        
        nr_of_iterations += 1

        if(f(a)*f(c) > 0):
            a = c
        if(f(b)*f(c) > 0):
            b = c

    return c, nr_of_iterations

def newton(f: typing.Callable[[float], float], df: typing.Callable[[float], float], ddf: typing.Callable[[float], float], a: Union[int,float], b: Union[int,float], epsilon: float, iteration: int) -> Tuple[float, int]:
    ''' Funkcja aproksymująca rozwiązanie równania f(x) = 0 metodą Newtona.
    Parametry: 
    f - funkcja dla której jest poszukiwane rozwiązanie
    df - pochodna funkcji dla której jest poszukiwane rozwiązanie
    ddf - druga pochodna funkcji dla której jest poszukiwane rozwiązanie
    a - początek przedziału
    b - koniec przedziału
    epsilon - tolerancja zera maszynowego (warunek stopu)
    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or not isinstance(epsilon, (int, float)) or not isinstance(iteration, int):
        return None
    
    if f(a)*f(b) > 0 or a >= b or df(a)*df(b) < 0 or ddf(a)*ddf(b) < 0:
        return None

    nr_of_iterations = 0

    if f(a) * ddf(a) > 0:
        x = a
    else:
        x = b 

    while nr_of_iterations < iteration:
        
        c = x - f(x)/df(x)

        nr_of_iterations += 1

        if np.abs(f(c)) < epsilon:
            return c, nr_of_iterations
        else:
            x = c

    return None
