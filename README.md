plots are still in progress, but some examples:

```python
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

plot4 = ggplot(
    mpg, aes(x="cty", fill="class")
) + geom_histogram() + scale_x_continuous()

mpg = mpg.copy(deep=True)
mpg["cty"] = mpg["cty"].astype(float)
plot5 = ggplot(mpg, aes(x="cty", y="displ", size="cyl", color="cty")) + geom_point()

ggmulti(
    [plot1, plot2, plot3, plot4, plot5],
    plots_path / "gg_multi_pmg.png",
)
```
<img src="plots/gg_multi_pmg.png?v=1" alt="gg_multi_pmg" width="800px">


```python
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = ggplot(mpg, aes("class")) + geom_bar()
    res = ggcreate(plot)
    ggdraw_plot(res, data_path / "geom_bar.png")
```
<img src="plots/geom_bar.png?v=1" alt="geom_bar" width="400px">


```python
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = ggplot(mpg, aes("class", fill="drv")) + geom_bar()
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_bar_fill.png")
```
<img src="plots/geom_bar_fill.png?v=1" alt="gg_point" width="400px">

```python
    rng = np.random.default_rng(42)

    x_vals = np.repeat(np.arange(28), 28)
    y_vals = np.tile(np.arange(28), 28)
    z_vals = rng.random(28 * 28)

    df = pd.DataFrame({
        'xs': x_vals.astype(float),
        'ys': y_vals.astype(float),
        'zs': z_vals
    })
    plot = ggplot(df, aes("xs", "ys", fill="zs")) + geom_tile()
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_tile.png")
```
<img src="plots/geom_tile.png?v=1" alt="geom_tile" width="400px">


```python
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
```
<img src="plots/geom_tile_mpg.png?v=1" alt="geom_tile" width="400px">


```python
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = (
        ggplot(mpg, aes(x="cty", color="class")) + geom_freqpoly() + scale_x_continuous()
    )
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_freqpoly.png")
```
<img src="plots/geom_freqpoly_cty_class.png?v=1" alt="gg_point" width="400px">

```python
diamonds = pd.read_csv(data_path / "diamonds.csv")
plot = ggplot(
    diamonds, aes("price", color="cut")
) + geom_freqpoly(
) + ylab(
    label="custom label",
    rotate=-45
) + xlab(
    rotate=45,
    tick_margin=2.5
)
res = ggcreate(plot)
ggdraw_plot(res, plots_path / "geom_freqpoly.png")
```
<img src="plots/geom_freqpoly.png?v=1" alt="gg_point" width="400px">

```python
mpg = pd.read_csv(data_path / "mpg.csv")
plot = (
    ggplot(mpg, aes(x="cty", fill="class"))
    + geom_freqpoly(alpha=0.3)
    + scale_x_continuous()
)
res = ggcreate(plot)
ggdraw_plot(res, plots_path / "test_geom_freqpoly_cty_class_fill.png")
```
<img src="plots/freqpoly_cty_class_fill.png?v=1" alt="gg_point" width="400px">


```python
mpg = pd.read_csv(data_path / "mpg.csv")
plot = ggplot(mpg, aes(x = 'displ')) + geom_histogram()
res = ggcreate(plot)
ggdraw_plot(res, data_path / "geom_histogram.png")
```
<img src="plots/geom_histogram.png?v=1" alt="geom_histogram" width="400px">

```python
mpg = pd.read_csv(data_path / "mpg.csv")
plot = ggplot(mpg, aes(x="cty", fill="class")) + geom_histogram() + scale_x_continuous()
res = ggcreate(plot)
ggdraw_plot(res, plots_path / "geom_histogram_fill.png")
```
<img src="plots/geom_histogram_fill.png?v=1" alt="geom_histogram" width="400px">


```python
    df = pd.DataFrame({
        'g': ['a', 'a', 'a', 'b', 'b', 'b'],
        'x': [1, 3, 5, 2, 4, 6],
        'y': [2, 5, 1, 3, 6, 7]
    })
    plot = ggplot(
        df, aes(x="x", y="y", fill="g")
    ) + geom_area(alpha=0.3) + geom_point(size=5)
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_area_stat_identity.png")
```
<img src="plots/geom_area_stat_identity.png?v=1" alt="geom_area_stat_identity" width="400px">


```python
    np.random.seed(1234)

    sex = np.repeat(["F", "M"], repeats=200)
    weight = np.round(
        np.concatenate(
            [
                np.random.normal(loc=55, scale=5, size=200),
                np.random.normal(loc=65, scale=5, size=200),
            ]
        )
    )

    df = pd.DataFrame({"sex": pd.Categorical(sex), "weight": weight})

    vline_gender_quantiles = (
        df.groupby("sex")["weight"].quantile([0.05, 0.95]).reset_index()
    )
    global_quantiles: List[float] = list(df["weight"].quantile([0.05, 0.95]))

    plot = (
        ggplot(df, aes(x="weight", fill="sex"))
        + geom_area(stat="bin", alpha=1)
        + geom_vline(
            data=vline_gender_quantiles,
            aes=aes(xintercept="weight"),
            size=2,
            line_type="dashed",
            inhert_aes=True,
            alpha=0.7,
        )
        + geom_vline(
            xintercept=global_quantiles, size=2.5, line_type="solid", color="blue"
        )
        + geom_hline(yintercept=10, size=1, alpha=0.7)
    )
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_area_stat_bin.png")
```
<img src="plots/geom_area_stat_bin.png?v=1" alt="geom_area_stat_bin" width="400px">


```python
    weather = pd.read_csv(data_path / "lincoln-weather.csv")

    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    weather['Month'] = pd.Categorical(weather['Month'], categories=month_order, ordered=True)

    plot = (
        ggplot(
            weather,
            aes(x="Mean Temperature [F]", fill="Month"),
        )
        + ggridges("Month", overlap=1.7)
        + geom_area(stat="bin", alpha=0.7)
        + ylab(rotate=-30)
    )

    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "ridgets_weather.png")
```
<img src="plots/ridgets_weather.png?v=1" alt="ridgets_weather" width="400px">


```python
mpg = pd.read_csv(data_path / "mpg.csv")
mpg["cty"] = mpg["cty"].astype(float)
plot = ggplot(mpg, aes(x="displ", y="hwy", color="cty")) + geom_point()
res = ggcreate(plot)
ggdraw_plot(res, plots_path / "geom_point_with_continuous_color.png")
```
<img src="plots/geom_point_with_continuous_color.png?v=1" alt="geom_point_with_continuous_color" width="400px">


```python
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
```
<img src="plots/geom_point_and_text.png?v=1" alt="geom_point_and_text" width="400px">


```python
    mt_cars = pd.read_csv(data_path / "mtcars_r.csv")

    plot = (
        ggplot(mt_cars, aes(x="wt", y="mpg"))
        + geom_point()
        + geom_abline(intercept=37, slope=-5, size=3.2)
        + geom_vline(xintercept=3, color="blue")
        + geom_hline(yintercept=22, line_type="dashed")
        + annotate_text("Annotated text", x=4, y=30, size=15, background_color="transparent")
        + annotate_text("🥸", x=3, y=22, size=40, emoji=True)
    )
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_abline_vline_hline.png")
```
<img src="plots/geom_abline_vline_hline.png?v=1" alt="geom_abline_vline_hline" width="400px">


```python
    mpg = pd.read_csv(data_path / "mpg.csv")
    mpg["hwy"] = mpg["hwy"].astype(float)
    plot = (
        ggplot(mpg, aes("displ", "hwy"))
        + geom_point(
            data=mpg.loc[mpg["manufacturer"] == "subaru"], color="orange", size=3
        )
        + geom_point(size=1.5)
        + annotate_curve(x=5, y=38, xend=3, yend=30, curvature=-0.3, arrow=True)
        + annotate_text(text="subaru", x=5, y=37, background_color="transparent")
        + annotate_point(x=4.95, y=36.3, color="orange", size=3)
        + annotate_point(x=4.95, y=36.3, color="black", size=1.5)
    )
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "annotate_curve.png")
```
<img src="plots/annotate_curve.png?v=1" alt="annotate_curve" width="400px">


```python
mpg = pd.read_csv(data_path / "mpg.csv")
mpg["cty"] = mpg["cty"].astype(float)
plot = ggplot(mpg, aes(x="cty", y="displ", size = "cyl", color="cty")) + geom_point()
res = ggcreate(plot)
ggdraw_plot(res, plots_path / "geom_point_with_continuous_color_and_size.png")
```
<img src="plots/geom_point_with_continuous_color_and_size.png?v=1" alt="geom_point_with_continuous_color_and_size" width="400px">


```python
    df = create_dataframe()
    plot = ggplot(
        df, aes("tenors", "pathValues", color = "pathNames")
    ) + geom_line() + xlab(rotate=-90, tick_margin=3)
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_line_With_color.png")
```
<img src="plots/geom_line_With_color.png?v=1" alt="geom_line_With_color" width="400px">


```python
df = pd.DataFrame(
    data={
        "trt": pd.Categorical([1, 1, 2, 2]),
        "resp": [1, 5, 3, 4],
        "group": pd.Categorical([1, 2, 1, 2]),
        "upper": [1.1, 5.3, 3.3, 4.2],
        "lower": [0.8, 4.6, 2.4, 3.6],
    }
)
plot = ggplot(df, aes(x="trt", y="resp", color="group")) + geom_linerange(
    aes(ymin="lower", ymax="upper")
)
res = ggcreate(plot)
ggdraw_plot(res, plots_path / "geom_linerange.png")
```
<img src="plots/geom_linerange.png?v=1" alt="geom_linerange" width="400px">

```python
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = ggplot(mpg, aes(x = 'displ', y = "cty", text = 'manufacturer')) + geom_text()
    res = ggcreate(plot)
    ggdraw_plot(res, data_path / "geom_text.png")
```
<img src="plots/geom_text.png?v=1" alt="geom_text" width="400px">


```python
    df = pd.DataFrame(
        data={"dose": ["D0.5", "D1", "D2"], "bbb": [4.2, 10, 29.5]}
    )
    plot = ggplot(df, aes(x="dose", y="bbb")) + geom_line() + geom_point()
    res = ggcreate(plot)
    ggdraw_plot(res, data_path / "geom_line_and_point_with_linetype.png")
```
<img src="plots/geom_line_and_point_with_linetype.png?v=1" alt="geom_line_and_point" width="400px">

```python
    df = pd.DataFrame({
        'x': [1, 3, 5, 2, 4, 6],
        'y': [2, 5, 1, 3, 6, 7]
    })
    plot = ggplot(
        df, aes(x="x", y="y")
    ) + geom_area(alpha=0.3)
    res = ggcreate(plot)
    ggdraw_plot(res, plots_path / "geom_area_simple.png")
```
<img src="plots/geom_area_simple.png?v=1" alt="geom_area_simple" width="400px">

```python
plots = _gg_multi_plots()
ggmulti(
    plots,
    plots_path / "gg_multi_pmg_bottom_to_top.png",
    vertical_orientation="bottom_to_top"
)
```
<img src="plots/gg_multi_pmg_bottom_to_top.png?v=1" alt="gg_multi_pmg_bottom_to_top" width="400px">

```python
plots = _gg_multi_plots()
ggmulti(
    plots,
    plots_path / "gg_multi_pmg_right_to_left.png",
    horizontal_orientation="right_to_left"
)
```
<img src="plots/gg_multi_pmg_right_to_left.png?v=1" alt="gg_multi_pmg_right_to_left" width="400px">


```python
df = pd.DataFrame({
    'trt': [1, 1, 2, 2],
    'resp': [1, 5, 3, 4],
    'group': pd.Categorical([1, 2, 1, 2]),
    'upper': [1.5, 5.0, 3.3, 4.2],
    'lower': [1, 4.0, 2.4, 3.6]
})
plot = ggplot(df, aes(x="trt", y="resp", color="group")) + geom_error_bar(
    aes(ymin="lower", ymax="upper"), size=20
)
res = ggcreate(plot)
ggdraw_plot(res, plots_path / "geom_error_bar.png")
```
<img src="plots/geom_error_bar.png?v=1" alt="geom_error_bar" width="400px">

![gg](plots/simple_test.png?v=1)
