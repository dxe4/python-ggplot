import pandas as pd
import pytest

from python_ggplot.public_interface.aes import aes
from python_ggplot.public_interface.common import ggdraw_plot, ggtitle
from python_ggplot.public_interface.geom import (
    geom_bar,
    geom_error_bar,
    geom_histogram,
    geom_line,
    geom_point,
    geom_text,
    ggplot,
)
from python_ggplot.public_interface.utils import ggcreate
from tests import data_path


def test_geom_bar():
    """
    this needs some more fixing.
    Y col is using X values
    """
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = ggplot(mpg, aes("class")) + geom_bar()
    res = ggcreate(plot)
    ggdraw_plot(res, data_path / "geom_bar.png")


def test_geom_point():
    mpg = pd.read_csv(data_path / "mpg.csv")  # type: ignore
    plot = (
        ggplot(mpg, aes(x="displ", y="hwy", color="class"))
        + geom_point()
        + ggtitle("gg plotting")
    )

    res = ggcreate(plot)
    ggdraw_plot(res, data_path / "geom_point.png")


def test_geom_point_with_color():
    mpg = pd.read_csv(data_path / "mpg.csv")  # type: ignore
    plot = ggplot(mpg, aes(x="displ", y="hwy")) + geom_point(
        aes(color="class"), size=3, alpha=0.7
    )
    # TODO labs needs to be implemented
    # + labs(
    #     title = "Engine Size vs Highway Mileage",
    #     x = "Engine Displacement (L)",
    #     y = "Highway Mileage (mpg)"
    # )
    res = ggcreate(plot)
    ggdraw_plot(res, data_path / "geom_point_with_color.png")


def test_geom_line():
    df = pd.DataFrame(data={"dose": ["D0.5", "D1", "D2"], "bbb": [4.2, 10, 29.5]})
    plot = ggplot(df, aes(x="dose", y="bbb")) + geom_line() + geom_point()
    res = ggcreate(plot)
    ggdraw_plot(res, data_path / "geom_line_and_point.png")


@pytest.mark.xfail(reason="fix merge of user styles")
def test_geom_line_with_linetype():
    df = pd.DataFrame(data={"dose": ["D0.5", "D1", "D2"], "bbb": [4.2, 10, 29.5]})
    plot = (
        ggplot(df, aes(x="dose", y="bbb"))
        + geom_line(line_type="dashed")
        + geom_point()
    )
    res = ggcreate(plot)
    ggdraw_plot(res, data_path / "geom_line_and_point_with_linetype.png")


# def test_geom_histogram():
#     mpg = pd.read_csv(data_path / "mpg.csv")  # type: ignore
#     plot = ggplot(mpg, aes(x = 'year')) + geom_histogram()
#     res = ggcreate(plot)
#     ggdraw_plot(res, data_path / "geom_histogram.png")


def test_geom_text():
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = ggplot(mpg, aes(x = 'displ', y = "cty", text = 'manufacturer')) + geom_text()
    res = ggcreate(plot)
    ggdraw_plot(res, data_path / "geom_text.png")


@pytest.mark.xfail(reason="KeyError: 'y_min'")
def test_geom_error_bar():
    df = pd.DataFrame(
        data={
            "team": ["A", "B", "C"],
            "mean": [7.5, 23, 13.75],
            "sd": [3.415650, 2.943920, 3.685557],
        }
    )
    df = df.assign(lower=df["mean"] - df["sd"], upper=df["mean"] + df["sd"])
    plot = ggplot(df, aes(x="team", y="mean")) + geom_error_bar(
        aes(ymin="lower", ymax="upper")
    )
    res = ggcreate(plot)
    ggdraw_plot(res, data_path / "geom_error.png")
