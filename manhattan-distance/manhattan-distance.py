import numpy as np

def manhattan_distance(x: list, y: list) -> float:
    """
    Returns the Manhattan distance as a Python float.
    """
    # Write code here
    res=0.0
    for i in range(len(x)):
        res+=np.abs(x[i]-y[i])
    return res