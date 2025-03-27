from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

from unicodeplots.backends.dtypes import ArrayLike, DataOps, Numeric, dtypes
from unicodeplots.canvas import BrailleCanvas
from unicodeplots.components import BorderBox
from unicodeplots.utils import Color, ColorType


class Lineplot:
    """
    A class for creating line plots with Unicode characters.
    """

    def __init__(
        self,
        *args,
        colors: Optional[Union[ColorType, List[ColorType]]] = None,
        show_axes: bool = False,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        border: Optional[str] = "",
        legend: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize a line plot with data and styling parameters.

        Args:
            *args: Data to plot in various formats
            show_axes: Plots x & y axis on the canvas.
            colors: Colors of plots
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            legend: Whether to show a legend
            border: Style of border to show ("single", "double", "ascii", "none")
            **kwargs: Styling parameters for canvas and box configuration
        """
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend
        self.border_style = border

        # Determine whether to show border based on decorative elements
        self.show_border = bool(title or xlabel or ylabel or legend or border)

        # Parse data from args first
        self.canvas = BrailleCanvas(**kwargs)
        self.datasets = self._parse_arguments(*args)
        self.min_x, self.max_x, self.min_y, self.max_y = self._compute_data_bounds()
        self.show_axes = show_axes
        self.colors = colors or [color for color in Color if color != Color.INVALID]
        self.auto_scale = kwargs.get("auto_scale", True)
        self.plot()

    def _validate_data(self, data: Iterable, name: str) -> List[Numeric]:
        """Validate that data is an iterable of numbers."""
        if not isinstance(data, Iterable) or isinstance(data, str):
            raise TypeError(f"{name} data must be an iterable (list, tuple, etc.) of numbers, got {type(data)}")

        # Fast path for numpy arrays
        if hasattr(data, "__array__") and hasattr(data, "dtype"):
            return data

        validated: List[Numeric] = []

        # NOTE: we just not be validating the input dtype and assume user is smart.
        for value in data:
            if not dtypes.is_numeric(value):
                raise TypeError(f"{name} values must be numbers (int, float, or numpy numeric type), got {type(value)}")
            validated.append(value)

        return validated

    def _process_dataset(
        self,
        x_raw: Sequence[Numeric],
        y_raw: Union[ArrayLike, Callable],
        dataset_index: int = 0,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Validates, potentially generates, and scales a single x, y dataset."""

        # 1. Validate X data
        validated_x = self._validate_data(x_raw, "X")

        # 2. Obtain raw Y data (either directly or by applying callable)
        actual_y_raw: ArrayLike
        if callable(y_raw):
            try:
                actual_y_raw = DataOps.apply(y_raw, validated_x)
            except Exception as e:
                raise ValueError(f"Error applying function to X data for dataset {dataset_index + 1}: {e}") from e

            validated_y = self._validate_data(actual_y_raw, f"Y (from function, dataset {dataset_index + 1})")
        else:
            actual_y_raw = y_raw  # type: ignore # Assuming y_raw is Sequence if not callable
            # Validate the provided Y data
            validated_y = self._validate_data(actual_y_raw, "Y")

        # 3. Check length consistency
        if len(validated_x) != len(validated_y):
            raise ValueError(f"X and Y data length mismatch for dataset {dataset_index + 1}: {len(validated_x)} vs {len(validated_y)}")

        scaled_x = DataOps.apply(self.canvas.xscale, validated_x)
        scaled_y = DataOps.apply(self.canvas.yscale, validated_y)

        return scaled_x, scaled_y

    def _parse_arguments(self, *args) -> List[Tuple[List[Union[float, int]], List[Union[float, int]]]]:
        """
        Parse arguments similar to matplotlib.pyplot.plot

        Supported formats:`
        - y_data only: [1, 2, 3] (x will be range(len(y)))
        - x_data, y_data: [1, 2, 3], [4, 5, 6]
        - x_data, callable: [1, 2, 3], lambda x: x**2
        - Multiple pairs: x1, y1, x2, y2, ... (y can be data or callable)

        Returns:
            List of (x_data, y_data) tuples, scaled and validated.
        """
        datasets: List[Tuple[List[Union[float, int]], List[Union[float, int]]]] = []

        if len(args) == 0:
            return datasets

        # Case 1: Single array/list - treat as y values
        if len(args) == 1:
            y_arg = args[0]
            # Validate y first to determine length for x range
            validated_y_for_len = self._validate_data(y_arg, "Y")
            x_raw = list(range(len(validated_y_for_len)))
            # We pass the original y_arg here, validation happens inside _process_dataset
            scaled_x, scaled_y = self._process_dataset(x_raw, y_arg)
            datasets.append((scaled_x, scaled_y))
            return datasets

        # Case 2: Pairs of x, y or x, callable
        if len(args) % 2 != 0:
            raise ValueError(f"After the first argument, arguments must come in pairs (x, y) or (x, callable). Got {len(args)} arguments.")

        for i in range(0, len(args), 2):
            x_arg = args[i]
            y_arg = args[i + 1]
            dataset_index = i // 2

            scaled_x, scaled_y = self._process_dataset(x_arg, y_arg, dataset_index=dataset_index)
            datasets.append((scaled_x, scaled_y))

        return datasets

    def _compute_data_bounds(self) -> tuple[Numeric, Numeric, Numeric, Numeric]:
        """
        Compute the min/max x and y values across all datasets.

        Returns:
            tuple: (min_x, max_x, min_y, max_y)
        """
        if not self.datasets:
            return (0.0, 1.0, 0.0, 1.0)

        min_x: Numeric = float("inf")
        max_x: Numeric = float("-inf")
        min_y: Numeric = float("inf")
        max_y: Numeric = float("-inf")

        # Initialize with the first point of the first dataset
        for x_data, y_data in self.datasets:
            min_x = min(min_x, DataOps.min(x_data))
            max_x = max(max_x, DataOps.max(x_data))
            min_y = min(min_y, DataOps.min(y_data))
            max_y = max(max_y, DataOps.max(y_data))

        # Prevent division by zero if all values are the same
        if min_x == max_x:
            min_x -= 0.5
            max_x += 0.5
        if min_y == max_y:
            min_y -= 0.5
            max_y += 0.5

        return (min_x, max_x, min_y, max_y)

    def plot(self):
        """
        Draw the plot on the canvas.

        Returns:
            self: For method chaining
        """
        if not self.datasets:
            return self

        # Update canvas parameters if auto_scale is enabled
        if self.auto_scale:
            # Set the origin point to match the data bounds
            self.canvas.params.origin_x = self.min_x
            self.canvas.params.origin_y = self.min_y
            # Update width and height to match data range
            self.canvas.params.width = self.max_x - self.min_x
            self.canvas.params.height = self.max_y - self.min_y

        # add xy axis
        if self.show_axes:
            x_axis_y = max(0, self.min_y) if 0 >= self.min_y and 0 <= self.max_y else self.min_y
            y_axis_x = max(0, self.min_x) if 0 >= self.min_x and 0 <= self.max_x else self.min_x

            self.canvas.line(self.min_x, x_axis_y, self.max_x, x_axis_y, color="WHITE")
            self.canvas.line(y_axis_x, self.min_y, y_axis_x, self.max_y, color="WHITE")

        for idx, (x_data, y_data) in enumerate(self.datasets):
            color = self.colors[idx % len(self.colors)]

            # Ensure there are enough points to form line segments
            if len(x_data) > 1:
                x_starts = x_data[:-1]
                y_starts = y_data[:-1]
                x_ends = x_data[1:]
                y_ends = y_data[1:]

                self.canvas.lines(x_starts, y_starts, x_ends, y_ends, color=color)

        return self

    def render(self) -> str:
        """
        Render the canvas to screen or return as string.

        Returns:
            str: The string representation of the plot
        """
        raw_plot = self.canvas.render()

        # If no decorative elements are requested, return the raw plot
        if not self.show_border:
            return raw_plot

        plot_lines = raw_plot.splitlines()

        border_box = BorderBox(width=self.canvas.rows, height=self.canvas.cols, border_type=self.border_style or "")

        if self.title:
            border_box.set_title(self.title)

        if self.xlabel:
            border_box.set_x_label(self.xlabel)

        if self.ylabel:
            border_box.set_y_label(self.ylabel)

        border_box.set_ranges((self.min_x, self.max_x), (self.min_y, self.max_y))

        # Add legend items if requested
        if self.legend and hasattr(self, "legend_items"):
            print("Note: This is not implemented yet")

        framed_plot = border_box.render(plot_lines)
        return "\n".join(framed_plot)
