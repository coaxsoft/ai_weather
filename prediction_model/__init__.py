"""Package contains helpful static functions for the model"""
import numpy
import pandas


def to_vector(s: pandas.Series) -> numpy.ndarray:
    """Convert Series into numpy vector with shape (N, 1)

    :param s: series to convert
    :return: converted vector
    """
    return numpy.array(s).reshape((len(s), 1))


def norma(v: numpy.ndarray) -> numpy.ndarray:
    """Normalize vector with respect to 1, i. e. ||v||"""
    s = v.sum()
    if s == 0:
        n = numpy.zeros(v.shape)
        n[0, 0] = 1
        return n
    return v / s


def reverse_norma(v: numpy.ndarray) -> numpy.ndarray:
    """Normalize vector with respect to 1 and reverse, i. e. || 1 - ||v|| ||"""
    return norma(1 - norma(v))


def max_or_null(x: numpy.ndarray) -> numpy.ndarray:
    """Generate new numpy array that contains one max element while
    all the other elements become 0
    """
    m = numpy.max(x)
    return numpy.array([0 if v < m else m for v in x]).reshape(x.shape)


def to_list(data: dict) -> dict:
    """Converts each element in data into ordinary list.
    **NOTE** all values in dict must be instances of ``np.ndarray`` class
    """
    return {k: v.reshape(v.size).tolist() for k, v in data.items()}


def none_to_zero(nda: numpy.ndarray, should_copy=False):
    """Change all ``NoneType`` in ``nda`` array.

    **NOTE** the function changes existing ``nda`` array, not
    creating new one by default. If you want co copy ``nda``,
    set ``should_copy`` argument to ``True``
    """
    if should_copy:
        nda = nda.copy()
    for i, _ in numpy.ndenumerate(nda):
        if nda[i] is None:
            nda[i] = 0
    return nda
