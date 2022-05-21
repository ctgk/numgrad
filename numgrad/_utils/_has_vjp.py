from numgrad._config import config


def has_vjp(forward_func: callable) -> bool:
    """Return if a forward function has VJP function registered or not.

    Parameters
    ----------
    forward_func : callable
        Forward function to check if it has VJP function registered.

    Returns
    -------
    bool
        True if the forward function has VJP registered else false.

    Example
    -------
    >>> has_vjp(np.random.normal)
    True
    >>> has_vjp(np.argsort)
    False
    """
    return forward_func in config._func2vjps
