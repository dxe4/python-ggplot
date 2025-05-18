from python_ggplot.public_interface.utils import fill_empty_spaces


def test_fill_grid_lr_tb():
    result = fill_empty_spaces(
        2,
        2,
        [1, 2, 3],
        horizontal_orientation="left_to_right",
        vertical_orientation="top_to_bottom",
    )
    assert result == [1, 2, 3, None]


def test_fill_grid_rl_tb():
    result = fill_empty_spaces(
        2,
        2,
        [1, 2, 3],
        horizontal_orientation="right_to_left",
        vertical_orientation="top_to_bottom",
    )
    assert result == [2, 1, None, 3]


def test_fill_grid_lr_bt():
    result = fill_empty_spaces(
        2,
        2,
        [1, 2, 3],
        horizontal_orientation="left_to_right",
        vertical_orientation="bottom_to_top",
    )
    assert result == [3, None, 1, 2]


def test_fill_grid_rl_bt():
    result = fill_empty_spaces(
        2,
        2,
        [1, 2, 3],
        horizontal_orientation="right_to_left",
        vertical_orientation="bottom_to_top",
    )
    assert result == [None, 3, 2, 1]


def test_fill_grid_lr_tb_3x3():
    result = fill_empty_spaces(
        3,
        3,
        [1, 2, 3, 4, 5, 6, 7],
        horizontal_orientation="left_to_right",
        vertical_orientation="top_to_bottom",
    )
    assert result == [1, 2, 3, 4, 5, 6, 7, None, None]


def test_fill_grid_rl_tb_3x3():
    result = fill_empty_spaces(
        3,
        3,
        [1, 2, 3, 4, 5, 6, 7],
        horizontal_orientation="right_to_left",
        vertical_orientation="top_to_bottom",
    )
    assert result == [3, 2, 1, 6, 5, 4, None, None, 7]


def test_fill_grid_lr_bt_3x3():
    result = fill_empty_spaces(
        3,
        3,
        [1, 2, 3, 4, 5, 6, 7],
        horizontal_orientation="left_to_right",
        vertical_orientation="bottom_to_top",
    )
    assert result == [7, None, None, 4, 5, 6, 1, 2, 3]


def test_fill_grid_rl_bt_3x3():
    result = fill_empty_spaces(
        3,
        3,
        [1, 2, 3, 4, 5, 6, 7],
        horizontal_orientation="right_to_left",
        vertical_orientation="bottom_to_top",
    )
    assert result == [None, None, 7, 6, 5, 4, 3, 2, 1]
