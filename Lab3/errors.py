import numpy as np
import scipy
import pickle

from typing import Union, List, Tuple


def absolut_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Obliczenie błędu bezwzględnego. 
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu bezwzględnego,
                                       NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(v, (int,float)):
        
        if isinstance(v_aprox, (int,float)):
            result =float(np.abs(v - v_aprox))
        
        elif isinstance(v_aprox, List):
            result =  [abs(v - x) for x in v_aprox]
        
        elif isinstance(v_aprox, np.ndarray):
            result = np.abs(v - v_aprox)
        
        else:
            return np.nan
   
    elif isinstance(v, List):
        
        if isinstance(v_aprox, (int,float)):
            result = [np.abs(x) - v_aprox for x in v]
        
        elif isinstance(v_aprox, List):
            
            if len(v) != len(v_aprox):
                return np.nan
            
            else:
                result = [np.abs(x - v_aprox[i]) for i, x in enumerate(v)]
                result = np.array(result)
        
        else:
            return np.nan
    
    elif isinstance(v, np.ndarray):
        
        if isinstance(v_aprox, np.ndarray):
            
            if np.shape(v)[0] != np.shape(v_aprox)[0]:
                return np.nan 
            
            else:
                result = np.abs(v - v_aprox)
        
        elif isinstance(v_aprox, (int,float)):
            result = np.abs(v - v_aprox)
        
        else:
            return np.nan
    
    else:
        return np.nan

    return result


def relative_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Obliczenie błędu względnego.
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu względnego,
                                       NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(v, (int,float)):
        
        if v == 0:
            return np.nan
        
        if isinstance(v_aprox, (int,float)):
            result = np.abs((v - v_aprox)/v)
        
        elif isinstance(v_aprox, List):
            result =  [abs((v - x)/v) for x in v_aprox]
        
        elif isinstance(v_aprox, np.ndarray):
            result = np.abs((v - v_aprox)/v)
        
        else:
            return np.nan

    elif isinstance(v, List):
        
        if isinstance(v_aprox, (int,float)):
            result = [np.abs(x- v_aprox)/x for x in v]
        
        elif isinstance(v_aprox, List):
            
            if len(v) != len(v_aprox):
                return np.nan
            
            else:
                result = [np.abs(x - v_aprox[i])/x for i, x in enumerate(v)]
        
        else:
            return np.nan
    
    elif isinstance(v, np.ndarray):
        
        if isinstance(v_aprox, np.ndarray):
            
            if np.shape(v)[0] != np.shape(v_aprox)[0]:
                return np.nan 
            
            else:
                result = np.abs((v - v_aprox)/v)
        
        elif isinstance(v_aprox, (int,float)):
            result = np.abs((v - v_aprox)/v)
        
        else:
            return np.nan
    
    else:
        return np.nan
    
    return result