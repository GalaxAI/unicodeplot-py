import math

import pytest

from unicodeplots import Lineplot

# --- Test Cases ---
# (test_id, args, output)
parse_arguments_cases = [
    ("no_args", (), []),
    # ("y_only", [1, 4, 9], ([1, 4, 9], [1, 4, 9])), # Doesn't pass curently. matplotlib allows. #TODO
    ("xy", ([1, 2, 3], [1, 2, 3]), [([1, 2, 3], [1, 2, 3])]),
    ("x_callable", ([1, 2, 3], lambda x: x * x), [([1, 2, 3], [1, 4, 9])]),
    (
        "alternating_mixed",
        (
            [1, 2],
            [10, 20],  # Dataset 1 (data, data)
            [3, 4, 6],
            lambda x: x * 10,  # Dataset 2 (data, callable)
        ),
        (([1, 2], [10, 20]), ([3, 4, 6], [30, 40, 60])),
    ),
]

# Structure:
# (description, x_range_params, y_generator_func, yscale_func, expected_scaled_min_y, expected_scaled_max_y)
# x_range_params could be (start, stop) for range()
# y_generator_func takes x and returns y
lineplot_data_and_scale_cases = [
    ("Linear scale, Y=2^X, X=[1..10]", (1, 11), lambda x: 2**x, None, 2, 1024),
    ("Log2 scale, Y=2^X, X=[1..10]", (1, 11), lambda x: 2**x, math.log2, 1, 10),
    ("Linear scale, Y=3*X, X=[0..5]", (0, 6), lambda x: 3 * x, None, 0, 15),
    ("Log10 scale, Y=10^X, X=[1..4]", (1, 5), lambda x: 10**x, math.log10, 1, 4),
    ("Linear scale, Y=X^2, X=[-3..3]", (-3, 4), lambda x: x**2, None, 0, 9),
    ("Log scale (base e), Y=e^X, X=[1..5]", (1, 6), lambda x: math.exp(x), math.log, 1.0, 5.0),
]


@pytest.mark.parametrize("test_id, args, expected_output", parse_arguments_cases, ids=[c[0] for c in parse_arguments_cases])
def test_parse_arguments(test_id, args, expected_output):
    """
    Tests the Lineplot._parse_arguments method with various input formats.
    Uses dummy scaling functions to isolate parsing logic.
    """
    # Create a minimal Lineplot instance
    # We don't need real data or bounds for this parsing test
    plot = Lineplot()

    # --- Call the method under test ---
    # We need to access the protected method (_parse_arguments)
    # pylint: disable=protected-access
    actual_output = plot._parse_arguments(*args)  # Unpack the args tuple

    # --- Assert ---
    for i, (actual_line, expected_line) in enumerate(zip(actual_output, expected_output)):
        assert len(actual_line) == 2, f"Actual line {i} data structure error"
        assert len(expected_line) == 2, f"Expected line {i} data structure error"

        # Compare the xy-coordinates list using pytest.approx
        assert actual_line[0] == pytest.approx(expected_line[0]), f"X-coordinates mismatch for line {i}"
        assert actual_line[1] == pytest.approx(expected_line[1]), f"Y-coordinates mismatch for line {i}"


@pytest.mark.parametrize(
    "description, x_range_params, y_generator_func, yscale_func, expected_scaled_min_y, expected_scaled_max_y",
    lineplot_data_and_scale_cases,
    ids=[case[0] for case in lineplot_data_and_scale_cases],
)
def test_lineplot_data_and_scale(description, x_range_params, y_generator_func, yscale_func, expected_scaled_min_y, expected_scaled_max_y):
    """
    Tests Lineplot with various data generation methods and y-axis scaling.
    """
    # 1. Generate Data based on parameters
    x_start, x_stop = x_range_params
    X_gen = list(range(x_start, x_stop))

    if not X_gen:
        pytest.skip(f"Skipping test '{description}' due to empty X data generation.")
        return  # For clarity

    Y_gen = [y_generator_func(x) for x in X_gen]

    # 2. Calculate Expected X Min/Max from generated data
    expected_min_x = min(X_gen)
    expected_max_x = max(X_gen)

    # 3. Instantiate Lineplot
    if yscale_func:
        plot = Lineplot(X_gen, Y_gen, yscale=yscale_func)
    else:
        # None * x  -> None and raises error
        plot = Lineplot(X_gen, Y_gen)

    # 4. Assertions
    # Assert X range (calculated from generated data)
    assert plot.min_x == expected_min_x, f"Failed min_x check for {description}"
    assert plot.max_x == expected_max_x, f"Failed max_x check for {description}"

    # Assert Y range *after scaling* (using provided expected values)
    # Use pytest.approx for float comparisons that might arise from scaling functions
    assert plot.min_y == pytest.approx(expected_scaled_min_y), f"Failed min_y (scaled) check for {description}"
    assert plot.max_y == pytest.approx(expected_scaled_max_y), f"Failed max_y (scaled) check for {description}"
