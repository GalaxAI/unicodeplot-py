import builtins
import collections.abc
from dataclasses import dataclass
from typing import Callable, Sequence, TypeAlias, Union

# Make numpy import optional
NUMPY_AVAILABLE = False
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    # Create a placeholder class for type hints when numpy isn't available
    class np:  # type: ignore # Allow redefining np for type hinting
        class ndarray:
            pass

        class number:
            pass


# --- Define TypeAliases at the module level ---
Numeric: TypeAlias = Union[int, float, "np.number"]
ArrayLike: TypeAlias = Union[Sequence[Numeric], "np.ndarray"]
ArrayLikeOrNumeric = Union[ArrayLike, Numeric]
CallableOrData: TypeAlias = Union[ArrayLike, Callable[[Numeric], Numeric]]


# --- Keep the dtypes class for helper methods ---
@dataclass
class dtypes:
    """Container class for utility functions related to data types."""

    @staticmethod
    def is_numeric(value) -> bool:
        """Check if a value is a numeric type (including numpy types if available)."""
        if isinstance(value, (int, float)):
            return True
        if NUMPY_AVAILABLE and isinstance(value, np.number):
            return True
        return False

    @staticmethod
    def to_native_type(value):
        """Convert numpy types to native Python types if needed."""
        if NUMPY_AVAILABLE and isinstance(value, np.number):
            return value.item()  # Convert numpy scalar to native Python type
        return value


class DataOps:
    """
    Provides data operations that adapt to the inpuit data type
    """

    @staticmethod
    def min(data: ArrayLike) -> Numeric:
        """Calculates the minimum value"""
        if len(data) == 0:
            raise ValueError("min() arg is an empty sequence")

        if NUMPY_AVAILABLE and isinstance(data, np.ndarray):
            return np.min(data)
        elif isinstance(data, collections.abc.Sequence) and not isinstance(data, str):
            return builtins.min(data)
        else:
            raise TypeError(f"Unsupported data type for DataOps.get_min: {type(data)}")

    @staticmethod
    def max(data: ArrayLike) -> Numeric:
        """Calculates the maximum value"""
        if len(data) == 0:
            raise ValueError("max() arg is an empty sequence")

        if NUMPY_AVAILABLE and isinstance(data, np.ndarray):
            return np.max(data)
        elif isinstance(data, collections.abc.Sequence) and not isinstance(data, str):
            return builtins.max(data)
        else:
            raise TypeError(f"Unsupported data type for DataOps.get_min: {type(data)}")

    @staticmethod
    def apply(func: Callable, data: ArrayLike) -> ArrayLike:
        """
        Applies a callable function element-wise to the data, preserving type
        (list -> list, ndarray -> ndarray) where possible.
        """
        if not callable(func):
            raise TypeError("`func` must be a callable.")

        if len(data) == 0:
            # Return empty container of the original type if possible
            if NUMPY_AVAILABLE and isinstance(data, np.ndarray):
                return np.array([])
            elif isinstance(data, collections.abc.Sequence) and not isinstance(data, str):
                return []

        if NUMPY_AVAILABLE and isinstance(data, np.ndarray):
            try:
                try:
                    return func(data)
                except TypeError:  # Fallback if func doesn't work directly on arrays
                    vectorized_func = np.vectorize(func)
                    return vectorized_func(data)
            except Exception as e:
                raise ValueError(f"Error applying function to numpy array: {e}") from e
        elif isinstance(data, collections.abc.Sequence) and not isinstance(data, str):
            try:
                return [func(x) for x in data]
            except Exception as e:
                raise ValueError(f"Error applying function to sequence: {e}") from e
        else:
            raise TypeError(f"Unsupported data type for DataOps.apply: {type(data)}")

    @staticmethod
    def round(data: ArrayLike) -> ArrayLike:
        """Rounds elements of the data to the nearest integer."""
        if NUMPY_AVAILABLE and isinstance(data, np.ndarray):
            return np.round(data)
        elif isinstance(data, collections.abc.Sequence) and not isinstance(data, str):
            return [round(x) for x in data]
        else:
            raise TypeError(f"Unsupported data type for DataOps.round: {type(data)}")

    @staticmethod
    def astype_int(data: ArrayLike) -> ArrayLike:
        """Converts elements of the data to integer type."""
        if NUMPY_AVAILABLE and isinstance(data, np.ndarray):
            return data.astype(np.int_)
        elif isinstance(data, collections.abc.Sequence) and not isinstance(data, str):
            return [builtins.int(x) for x in data]
        else:
            raise TypeError(f"Unsupported data type for DataOps.astype_int: {type(data)}")
