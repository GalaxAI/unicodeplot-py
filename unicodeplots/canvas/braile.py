from typing import Optional

from unicodeplots.backends.dtypes import ArrayLike, DataOps, Numeric
from unicodeplots.canvas.canvas import Canvas
from unicodeplots.utils import CanvasParams, Color, ColorType

SUPERSAMPLE = 8  # Note: This helps graph look better when there is small number of data points.


class BrailleCanvas(Canvas):
    @property
    def x_pixel_per_char(self) -> int:
        return 2

    @property
    def y_pixel_per_char(self) -> int:
        return 4

    def __init__(self, params: Optional[CanvasParams] = None, **kwargs):
        """
        Create a Braille-based canvas for unicode plotting.

        Args:
            params: Canvas parameters as a CanvasParams object
            **kwargs: Additional parameters that override values in params if provided
        """
        super().__init__(params=params, **kwargs)

        self.default_char = 0x2800
        self.active_cells = [[self.default_char] * self.grid_cols for _ in range(self.grid_rows)]

        self.default_color = Color.WHITE
        self.active_colors = [[self.default_color] * self.grid_cols for _ in range(self.grid_rows)]

        # Precompute bit values for faster pixel calculations
        self.bit_table = [
            [0x01, 0x02, 0x04, 0x40],  # x=0
            [0x08, 0x10, 0x20, 0x80],  # x=1
        ]

    def _set_pixel(self, px: int, py: int, color: ColorType) -> None:
        """Set a pixel in the Braille grid representation."""
        cx = px // self.x_pixel_per_char
        cy = py // self.y_pixel_per_char

        if not (0 <= cx < self.grid_cols and 0 <= cy < self.grid_rows):
            return

        try:
            x_in = px % self.x_pixel_per_char
            y_in = py % self.y_pixel_per_char
            bit = self.bit_table[x_in][y_in]
        except IndexError:
            return

        # Update cell bits
        self.active_cells[cy][cx] |= bit

        # Update color (simple overwrite)
        self.active_colors[cy][cx] = color

    def _draw_bresenham_segment(self, px1: int, py1: int, px2: int, py2: int, color: ColorType):
        """Draws a single line segment using Bresenham given INTEGER pixel coordinates."""

        dx = abs(px2 - px1)
        dy = abs(py2 - py1)
        sx = 1 if px1 < px2 else -1
        sy = 1 if py1 < py2 else -1
        err = dx - dy

        pixels = set()
        px_curr, py_curr = px1, py1

        while True:
            # Add the non-supersampled pixel coordinate
            pixels.add((px_curr // SUPERSAMPLE, py_curr // SUPERSAMPLE))

            if px_curr == px2 and py_curr == py2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                px_curr += sx
            if e2 < dx:
                err += dx
                py_curr += sy

        # Set the actual pixels on the canvas
        for p_x, p_y in pixels:
            self._set_pixel(p_x, p_y, color)

    def vec_line(self, x1_arr: ArrayLike, y1_arr: ArrayLike, x2_arr: ArrayLike, y2_arr: ArrayLike, color: ColorType):
        num_segments = len(x1_arr)
        px1_arr = self.x_to_pixel(x1_arr) * SUPERSAMPLE
        py1_arr = self.y_to_pixel(y1_arr) * SUPERSAMPLE
        px2_arr = self.x_to_pixel(x2_arr) * SUPERSAMPLE
        py2_arr = self.y_to_pixel(y2_arr) * SUPERSAMPLE

        # Vectorized rounding and conversion to integer arrays
        px1_i_arr = DataOps.astype_int(DataOps.round(px1_arr))
        py1_i_arr = DataOps.astype_int(DataOps.round(py1_arr))
        px2_i_arr = DataOps.astype_int(DataOps.round(px2_arr))
        py2_i_arr = DataOps.astype_int(DataOps.round(py2_arr))

        # Loop through calculated INTEGER coordinates and draw segments
        for i in range(num_segments):
            self._draw_bresenham_segment(px1_i_arr[i], py1_i_arr[i], px2_i_arr[i], py2_i_arr[i], color)

    def line(self, x1: Numeric, y1: Numeric, x2: Numeric, y2: Numeric, color: ColorType):
        """Draw a line between logical coordinates using supersampled Bresenham for smoother curves"""
        # Convert to high-resolution pixel coordinates (8x supersampling)
        # NOTE: If this really gives bad performance we can only do super sampling for curves or as a flag
        # Or go back to d34b commit
        px1 = self.x_to_pixel(x1) * SUPERSAMPLE
        py1 = self.y_to_pixel(y1) * SUPERSAMPLE
        px2 = self.x_to_pixel(x2) * SUPERSAMPLE
        py2 = self.y_to_pixel(y2) * SUPERSAMPLE

        # Standard Bresenham at high resolution
        px1, py1 = int(round(px1)), int(round(py1))
        px2, py2 = int(round(px2)), int(round(py2))

        self._draw_bresenham_segment(px1, py1, px2, py2, color)

    def lines(self, x_starts: ArrayLike, y_starts: ArrayLike, x_ends: ArrayLike, y_ends: ArrayLike, color: ColorType):
        """
        Draw multiple line segments efficiently.

        Accepts sequences (like lists or arrays) of coordinates.
        """
        num_segments = len(x_starts)
        if not (len(y_starts) == num_segments and len(x_ends) == num_segments and len(y_ends) == num_segments):
            raise ValueError("Input sequences for 'lines' must all have the same length")

        if hasattr(x_starts, "__array__") and hasattr(x_ends, "dtype"):
            print("Using vec_line")
            self.vec_line(x_starts, y_starts, x_ends, y_ends, color)
        else:
            # Scalar Path
            for x1, y1, x2, y2 in zip(x_starts, y_starts, x_ends, y_ends):
                self.line(x1, y1, x2, y2, color)

    def render(self) -> str:
        """Efficient rendering with pre-allocated strings"""
        return "\n".join(
            "".join(ColorType(self.active_colors[row][col]).apply(chr(self.active_cells[row][col])) for col in range(self.grid_cols))
            for row in range(self.grid_rows)
        )


# Demo Script
if __name__ == "__main__":
    canvas = BrailleCanvas(width=32, height=16, resolution=4, origin_x=0, origin_y=0, xflip=False, yflip=False, blend=False)

    points = [(0, 16), (5, 0), (32, 16)]

    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        print(f"Drawing line from {(x1, y1)} to {(x2, y2)}")
        canvas.line(x1, y1, x2, y2, color=Color.BLUE)
    b = [(0, 0), (5, 5), (32, 10)]
    for i in range(len(b) - 1):
        x1, y1 = b[i]
        x2, y2 = b[i + 1]
        print(f"Drawing line from {(x1, y1)} to {(x2, y2)}")
        canvas.line(x1, y1, x2, y2, color=Color.RED)

    print(canvas.render())
