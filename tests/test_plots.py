import pandas as pd
import pytest

from python_ggplot.public_interface.aes import aes
from python_ggplot.public_interface.common import (
    ggdraw_plot,
    ggtitle,
    scale_x_continuous,
)
from python_ggplot.public_interface.geom import (
    geom_bar,
    geom_error_bar,
    geom_freqpoly,
    geom_histogram,
    geom_line,
    geom_linerange,
    geom_point,
    geom_text,
    ggplot,
)
from python_ggplot.public_interface.utils import ggcreate
from tests import data_path, plots_path


def test_geom_bar():
    """
    this needs some more fixing.
    Y col is using X values
    """
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = ggplot(mpg, aes("class")) + geom_bar()
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_bar.png")


@pytest.mark.xfail(reason="fix")
def test_geom_point_and_text():
    """
    There's a few issues with this
    1) python_ggplot/gg/drawing.GetXY.calculate doesn't deal with y str
    2) multiple aes y cols conflict (the original here would have been a formula node)
    """
    mpg = pd.read_csv(data_path / "mpg.csv")

    mpg["mpgMean"] = (mpg["cty"] + mpg["hwy"]) / 2.0
    df_max = mpg.sort_values("mpgMean").tail(1)

    plot = (
        ggplot(mpg, aes("hwy", "class"))
        + geom_point(aes(color="cty"))
        + geom_text(data=df_max, aes=aes(y="displ", text="model"))
        + geom_text(data=df_max, aes=aes(y="displ", text="mpgMean"))
    )
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_point_and_text.png")


@pytest.mark.xfail(reason="x and y flipped for now (intentional)")
def test_geom_bar_y():
    """
    this produces the plot but it doesnt flip the axis for now
    post_process.get_scales flips the axis
    post_process needs an overall clean up, but the plan is to getting working first

    Flip the test_geom_bar
    Third plot here https://ggplot2.tidyverse.org/reference/geom_bar.html
    """

    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = ggplot(mpg, aes(y="class")) + geom_bar()
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_bar_y.png")


def test_geom_freq_poly():
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = (
        ggplot(mpg, aes(x="cty", color="class")) + geom_freqpoly() + scale_x_continuous()
    )
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_freqpoly_cty_class.png")


def test_geom_histogram_fill():
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = ggplot(mpg, aes(x="cty", fill="class")) + geom_histogram() + scale_x_continuous()
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_histogram_fill.png")


@pytest.mark.xfail(reason="")
def test_geom_bar_fill():
    # Fill the bars
    # Fourth plot here https://ggplot2.tidyverse.org/reference/geom_bar.html
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = ggplot(mpg, aes(fill="drv")) + geom_bar()
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_bar_fill.png")


def test_geom_point():
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = (
        ggplot(mpg, aes(x="displ", y="hwy", color="class"))
        + geom_point()
        + ggtitle("gg plotting")
    )

    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_point.png")


def test_geom_point_with_color():
    mpg = pd.read_csv(data_path / "mpg.csv")
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
    ggdraw_plot(res, plots_path / "geom_point_with_color.png")


def test_geom_line():
    df = pd.DataFrame(data={"dose": ["D0.5", "D1", "D2"], "bbb": [4.2, 10, 29.5]})
    plot = ggplot(df, aes(x="dose", y="bbb")) + geom_line() + geom_point()
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_line_and_point.png")


def test_geom_line_with_linetype():
    df = pd.DataFrame(data={"dose": ["D0.5", "D1", "D2"], "bbb": [4.2, 10, 29.5]})
    plot = (
        ggplot(df, aes(x="dose", y="bbb"))
        + geom_line(line_type="dashed")
        + geom_point()
    )
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_line_and_point_with_linetype.png")


def test_geom_histogram():
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = ggplot(mpg, aes(x="displ")) + geom_histogram()
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_histogram.png")


def test_geom_text():
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = ggplot(mpg, aes(x="displ", y="cty", text="manufacturer")) + geom_text()
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_text.png")


@pytest.mark.xfail(reason="incorrect plot")
def test_geom_error_bar():
    """
    this needs some further fixing
    """
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
    ggdraw_plot(res, plots_path / "geom_error_bar.png")


@pytest.mark.xfail(reason="incorrect plot")
def test_geom_linerange():
    df = pd.DataFrame(
        data={
            "trt": pd.Categorical([1, 1, 2, 2]),
            "resp": [1, 5, 3, 4],
            "group": pd.Categorical([1, 2, 1, 2]),
            "upper": [1.1, 5.3, 3.3, 4.2],
            "lower": [0.8, 4.6, 2.4, 3.6],
        }
    )
    plot = ggplot(df, aes(x="trt", y="resp")) + geom_linerange(
        aes(ymin="lower", ymax="upper")
    )
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_linerange.png")


def test_geom_freqpoly():
    diamonds = pd.read_csv(data_path / "diamonds.csv")
    plot = ggplot(diamonds, aes("price", color="cut")) + geom_freqpoly()
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_freqpoly.png")
