import numpy as np
import pandas as pd
import pytest

from python_ggplot.gg.types import gg_col
from python_ggplot.public_interface.aes import aes
from python_ggplot.public_interface.common import (
    draw_layout,
    ggdraw,
    ggdraw_plot,
    ggtitle,
    scale_x_continuous,
    scale_x_discrete,
    scale_y_continuous,
    scale_y_discrete,
    xlab,
    ylab,
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
    geom_tile,
    ggplot,
)
from python_ggplot.public_interface.utils import ggcreate, ggmulti, plot_layout
from tests import data_path, plots_path


def test_gg_multi_mpg():
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot1 = ggplot(mpg, aes("class", fill="drv")) + geom_bar()

    df = mpg.groupby(["class", "cyl"], as_index=False).agg(meanHwy=("hwy", "mean"))
    plot2 = (
        ggplot(df, aes("class", "cyl", fill="meanHwy"))
        + geom_tile()
        + geom_text(aes(text="meanHwy"))
        + scale_y_discrete()
    )

    plot3 = (
        ggplot(mpg, aes(x="cty", fill="class"))
        + geom_freqpoly(alpha=0.3)
        + scale_x_continuous()
    )

    plot4 = (
        ggplot(mpg, aes(x="cty", fill="class"))
        + geom_histogram()
        + scale_x_continuous()
    )

    mpg = mpg.copy(deep=True)
    mpg["cty"] = mpg["cty"].astype(float)
    plot5 = ggplot(mpg, aes(x="cty", y="displ", size="cyl", color="cty")) + geom_point()

    ggmulti(
        [plot1, plot2, plot3, plot4, plot5],
        plots_path / "gg_multi_pmg.png",
    )


def test_gg_multi_mpg_set_cols():
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot1 = ggplot(mpg, aes("class", fill="drv")) + geom_bar()

    df = mpg.groupby(["class", "cyl"], as_index=False).agg(meanHwy=("hwy", "mean"))
    plot2 = (
        ggplot(df, aes("class", "cyl", fill="meanHwy"))
        + geom_tile()
        + geom_text(aes(text="meanHwy"))
        + scale_y_discrete()
    )

    plot3 = (
        ggplot(mpg, aes(x="cty", fill="class"))
        + geom_freqpoly(alpha=0.3)
        + scale_x_continuous()
    )

    plot4 = (
        ggplot(mpg, aes(x="cty", fill="class"))
        + geom_histogram()
        + scale_x_continuous()
    )

    mpg = mpg.copy(deep=True)
    mpg["cty"] = mpg["cty"].astype(float)
    plot5 = ggplot(mpg, aes(x="cty", y="displ", size="cyl", color="cty")) + geom_point()

    ggmulti(
        [plot1, plot2, plot3, plot4, plot5],
        plots_path / "gg_multi_pmg_set_cols1.png",
        cols=3,
    )
    ggmulti(
        [plot1, plot2, plot3, plot4, plot5],
        plots_path / "gg_multi_pmg_set_cols2.png",
        cols=0,
    )
    ggmulti(
        [plot1, plot2, plot3, plot4, plot5],
        plots_path / "gg_multi_pmg_set_cols3.png",
        cols=-1,
    )
    ggmulti(
        [plot1, plot2, plot3, plot4, plot5],
        plots_path / "gg_multi_pmg_set_cols4.png",
        cols=3,
        empty_plots=4,
    )


def test_gg_multi_mpg_set_rows():
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot1 = ggplot(mpg, aes("class", fill="drv")) + geom_bar()

    df = mpg.groupby(["class", "cyl"], as_index=False).agg(meanHwy=("hwy", "mean"))
    plot2 = (
        ggplot(df, aes("class", "cyl", fill="meanHwy"))
        + geom_tile()
        + geom_text(aes(text="meanHwy"))
        + scale_y_discrete()
    )

    plot3 = (
        ggplot(mpg, aes(x="cty", fill="class"))
        + geom_freqpoly(alpha=0.3)
        + scale_x_continuous()
    )

    ggmulti([plot1, plot2, plot3], plots_path / "gg_multi_pmg_set_rows1.png", rows=3)
    ggmulti([plot1, plot2, plot3], plots_path / "gg_multi_pmg_set_rows2.png", rows=0)
    ggmulti([plot1, plot2, plot3], plots_path / "gg_multi_pmg_set_rows3.png", rows=-1)
    ggmulti(
        [plot1, plot2, plot3],
        plots_path / "gg_multi_pmg_set_rows4.png",
        rows=2,
        empty_plots=1,
    )


def test_gg_multi_mpg_set_rows_and_cols():
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot1 = ggplot(mpg, aes("class", fill="drv")) + geom_bar()

    df = mpg.groupby(["class", "cyl"], as_index=False).agg(meanHwy=("hwy", "mean"))
    plot2 = (
        ggplot(df, aes("class", "cyl", fill="meanHwy"))
        + geom_tile()
        + geom_text(aes(text="meanHwy"))
        + scale_y_discrete()
    )

    plot3 = (
        ggplot(mpg, aes(x="cty", fill="class"))
        + geom_freqpoly(alpha=0.3)
        + scale_x_continuous()
    )

    ggmulti(
        [plot1, plot2, plot3],
        plots_path / "gg_multi_pmg_set_rows_and_cols1.png",
        rows=2,
        cols=2,
    )
    ggmulti(
        [plot1, plot2, plot3],
        plots_path / "gg_multi_pmg_set_rows_and_cols2.png",
        rows=2,
        cols=2,
        empty_plots=2,
    )
    ggmulti(
        [plot1, plot2, plot3],
        plots_path / "gg_multi_pmg_set_rows_and_cols3.png",
        rows=3,
        cols=0,
    )
    ggmulti(
        [plot1, plot2, plot3],
        plots_path / "gg_multi_pmg_set_rows_and_cols4.png",
        rows=0,
        cols=3,
    )
    ggmulti(
        [plot1, plot2, plot3],
        plots_path / "gg_multi_pmg_set_rows_and_cols5.png",
        rows=0,
        cols=0,
    )
    ggmulti(
        [plot1, plot2, plot3],
        plots_path / "gg_multi_pmg_set_rows_and_cols6.png",
        rows=-1,
        cols=-1,
    )

    plot4 = (
        ggplot(mpg, aes(x="cty", fill="class"))
        + geom_histogram()
        + scale_x_continuous()
    )

    mpg = mpg.copy(deep=True)
    mpg["cty"] = mpg["cty"].astype(float)
    plot5 = ggplot(mpg, aes(x="cty", y="displ", size="cyl", color="cty")) + geom_point()
    ggmulti(
        [plot1, plot2, plot3, plot4, plot5],
        plots_path / "gg_multi_pmg_set_rows_and_cols7.png",
        rows=3,
        cols=3,
        empty_plots=[1, 3],
    )


def test_gg_multi_mpg_with_one_empty():
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot1 = ggplot(mpg, aes("class", fill="drv")) + geom_bar()

    df = mpg.groupby(["class", "cyl"], as_index=False).agg(meanHwy=("hwy", "mean"))
    plot2 = (
        ggplot(df, aes("class", "cyl", fill="meanHwy"))
        + geom_tile()
        + geom_text(aes(text="meanHwy"))
        + scale_y_discrete()
    )

    plot3 = (
        ggplot(mpg, aes(x="cty", fill="class"))
        + geom_freqpoly(alpha=0.3)
        + scale_x_continuous()
    )

    plot4 = (
        ggplot(mpg, aes(x="cty", fill="class"))
        + geom_histogram()
        + scale_x_continuous()
    )

    mpg = mpg.copy(deep=True)
    mpg["cty"] = mpg["cty"].astype(float)
    plot5 = ggplot(mpg, aes(x="cty", y="displ", size="cyl", color="cty")) + geom_point()

    ggmulti(
        [plot1, plot2, plot3, plot4, plot5],
        plots_path / "gg_multi_pmg_with_one_empty.png",
        empty_plots=4,
    )


def test_gg_multi_mpg_with_many_empty():
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot1 = ggplot(mpg, aes("class", fill="drv")) + geom_bar()

    df = mpg.groupby(["class", "cyl"], as_index=False).agg(meanHwy=("hwy", "mean"))
    plot2 = (
        ggplot(df, aes("class", "cyl", fill="meanHwy"))
        + geom_tile()
        + geom_text(aes(text="meanHwy"))
        + scale_y_discrete()
    )

    plot3 = (
        ggplot(mpg, aes(x="cty", fill="class"))
        + geom_freqpoly(alpha=0.3)
        + scale_x_continuous()
    )

    plot4 = (
        ggplot(mpg, aes(x="cty", fill="class"))
        + geom_histogram()
        + scale_x_continuous()
    )

    mpg = mpg.copy(deep=True)
    mpg["cty"] = mpg["cty"].astype(float)
    plot5 = ggplot(mpg, aes(x="cty", y="displ", size="cyl", color="cty")) + geom_point()

    ggmulti(
        [plot1, plot2, plot3, plot4, plot5, plot2, plot3],
        plots_path / "gg_multi_pmg_with_many_empty1.png",
        empty_plots=[3, 5],
    )

    ggmulti(
        [plot1, plot2, plot3, plot4],
        plots_path / "gg_multi_pmg_with_many_empty2.png",
        empty_plots=[2, 3],
        cols=3,
    )

    # # Another one that failed but can't work out why. Getting an IndexError
    # ggmulti(
    #     [plot1, plot2, plot3, plot4, plot5],
    #     plots_path / "gg_multi_pmg_with_many_empty3.png",
    #     empty_plots=[2, 3],
    #     cols=4
    # )

    # This one failed as we have picked 0 as one of the empty plots
    # and in `background_style = img.get_current_background_style()`
    # we get a """IndexError: list index out of range""" error
    # ggmulti(
    #     [plot1, plot2, plot3, plot4, plot5, plot2, plot3],
    #     plots_path / "gg_multi_pmg_with_many_empty4.png",
    #     empty_plots=[0,8]
    # )


def test_geom_bar():
    """
    this needs some more fixing.
    Y col is using X values
    """
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = ggplot(mpg, aes("class")) + geom_bar()
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_bar.png")


def test_geom_bar_fill():
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = ggplot(mpg, aes("class", fill="drv")) + geom_bar()
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_bar_fill.png")


def test_geom_point_and_text():
    mpg = pd.read_csv(data_path / "mpg.csv")

    mpg["cty"] = mpg["cty"].astype(float)
    mpg["mpgMean"] = (mpg["cty"] + mpg["hwy"]) / 2.0
    df_max = mpg.sort_values("mpgMean").tail(1)

    plot = (
        ggplot(mpg, aes("hwy", "displ"))
        + geom_point(aes(color="cty"))
        + geom_text(data=df_max, aes=aes(y=gg_col("displ") + 0.2, text="model"))
        + geom_text(data=df_max, aes=aes(y=gg_col("displ") - 0.2, text="mpgMean"))
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


def test_geom_histogram_fill():
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = (
        ggplot(mpg, aes(x="cty", fill="class"))
        + geom_histogram()
        + scale_x_continuous()
    )
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_histogram_fill.png")


@pytest.mark.xfail(reason="")
def test_geom_bar_fill_y_only():
    # Fill the bars
    # Fourth plot here https://ggplot2.tidyverse.org/reference/geom_bar.html
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = ggplot(mpg, aes(y="class")) + geom_bar(aes(fill="drv"))
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
    # TODO this needs fixing
    mpg = pd.read_csv(data_path / "mpg.csv")
    mpg["cty"] = mpg["cty"].astype(float)
    plot = ggplot(mpg, aes(x="displ", y="hwy", color="cty")) + geom_point()
    # TODO labs needs to be implemented
    # + labs(
    #     title = "Engine Size vs Highway Mileage",
    #     x = "Engine Displacement (L)",
    #     y = "Highway Mileage (mpg)"
    # )
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_point_with_continuous_color.png")


def test_geom_point_with_color_and_size():
    mpg = pd.read_csv(data_path / "mpg.csv")
    mpg["cty"] = mpg["cty"].astype(float)
    plot = ggplot(mpg, aes(x="cty", y="displ", size="cyl", color="cty")) + geom_point()
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_point_with_continuous_color_and_size.png")


def test_geom_tile():
    rng = np.random.default_rng(42)

    x_vals = np.repeat(np.arange(28), 28)
    y_vals = np.tile(np.arange(28), 28)
    z_vals = rng.random(28 * 28)

    df = pd.DataFrame(
        {"xs": x_vals.astype(float), "ys": y_vals.astype(float), "zs": z_vals}
    )
    plot = ggplot(df, aes("xs", "ys", fill="zs")) + geom_tile()
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_tile.png")


def test_geom_tile_mpg():
    mpg = pd.read_csv(data_path / "mpg.csv")

    df = mpg.groupby(["class", "cyl"], as_index=False).agg(meanHwy=("hwy", "mean"))

    plot = (
        ggplot(df, aes("class", "cyl", fill="meanHwy"))
        + geom_tile()
        + geom_text(aes(text="meanHwy"))
        + scale_y_discrete()
    )
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_tile_mpg.png")


def test_geom_line_and_point():
    df = pd.DataFrame(data={"dose": ["D0.5", "D1", "D2"], "bbb": [4.2, 10, 29.5]})
    plot = ggplot(df, aes(x="dose", y="bbb")) + geom_line() + geom_point()
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_line_and_point.png")


def test_geom_line_With_color():
    def create_dataframe(paths=10, dates=50, sigma=0.10, seed=124325):
        np.random.seed(seed)

        tenors = []
        path_names = []
        path_values = []

        for j in range(paths):
            values = [100.0]
            for i in range(1, dates):
                gaussian = np.random.normal(0.0, 1.0)
                next_value = values[-1] * np.exp(-0.5 * sigma**2 + sigma * gaussian)
                values.append(next_value)

            path_values.extend(values)
            path_names.extend([f"path{j+1}"] * dates)
            tenors.extend(range(dates))

        df = pd.DataFrame(
            {"tenors": tenors, "pathNames": path_names, "pathValues": path_values}
        )

        return df

    df = create_dataframe()
    plot = (
        ggplot(df, aes("tenors", "pathValues", color="pathNames"))
        + geom_line()
        + xlab(rotate=-90, tick_margin=3)
    )
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_line_With_color.png")


def test_geom_line_with_linetype():
    df = pd.DataFrame(data={"dose": ["D0.5", "D1", "D2"], "bbb": [4.2, 10, 29.5]})
    plot = (
        ggplot(df, aes(x="dose", y="bbb"))
        + geom_line(line_type="dashed")
        + geom_point()
    )
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_line_and_point_with_linetype.png")


def test_geom_histogram_simple():
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = ggplot(mpg, aes(x="displ")) + geom_histogram()
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_histogram.png")


def test_geom_text():
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = ggplot(mpg, aes(x="displ", y="cty", text="manufacturer")) + geom_text()
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_text.png")


def test_geom_error_bar():
    """
    this needs some further fixing
    """
    df = pd.DataFrame(
        {
            "trt": [1, 1, 2, 2],
            "resp": [1, 5, 3, 4],
            "group": pd.Categorical([1, 2, 1, 2]),
            "upper": [1.5, 5.0, 3.3, 4.2],
            "lower": [1, 4.0, 2.4, 3.6],
        }
    )
    plot = ggplot(df, aes(x="trt", y="resp", color="group")) + geom_error_bar(
        aes(ymin="lower", ymax="upper"), size=20
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


def test_geom_freqpoly_diamonds():
    diamonds = pd.read_csv(data_path / "diamonds.csv")
    plot = (
        ggplot(diamonds, aes("price", color="cut"))
        + geom_freqpoly()
        + ylab(label="custom label", rotate=-45)
        + xlab(rotate=45, tick_margin=2.5)
    )
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_freqpoly.png")


def test_geom_freqpoly_cty_class():
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = (
        ggplot(mpg, aes(x="cty", color="class"))
        + geom_freqpoly()
        + scale_x_continuous()
    )
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_freqpoly_cty_class.png")


def test_geom_freqpoly_cty_class_fill():
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = (
        ggplot(mpg, aes(x="cty", fill="class"))
        + geom_freqpoly(alpha=0.3)
        + scale_x_continuous()
    )
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "freqpoly_cty_class_fill.png")
