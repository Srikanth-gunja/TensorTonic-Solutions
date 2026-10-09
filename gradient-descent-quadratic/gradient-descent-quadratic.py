def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    # Write code here
    for _ in range(steps):
        x0=x0-lr*get_derviate(a,b,x0)
    return x0
def get_derviate(a,b,x):
    # 2ax+b
    return 2*a*x+b