# unicodeplots/backends/dtypes.py
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
