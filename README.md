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
        data={"dose": ["D0.5", "D1", "D2"], "bbb": [4.2, 10, 29.5]}
    )
    plot = ggplot(df, aes(x="dose", y="bbb")) + geom_line() + geom_point()
    res = ggcreate(plot)
    ggdraw_plot(res, data_path / "geom_line_and_point_with_linetype.png")
```
<img src="plots/geom_line_and_point_with_linetype.png?v=1" alt="geom_line_and_point" width="400px">

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
plot = ggplot(mpg, aes(x="cty", y="displ", size = "cyl", color="cty")) + geom_point()
res = ggcreate(plot)
ggdraw_plot(res, plots_path / "geom_point_with_continuous_color_and_size.png")
```
<img src="plots/geom_point_with_continuous_color_and_size.png?v=1" alt="geom_point_with_continuous_color_and_size" width="400px">



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


```python
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = ggplot(mpg, aes(x = 'displ', y = "cty", text = 'manufacturer')) + geom_text()
    res = ggcreate(plot)
    ggdraw_plot(res, data_path / "geom_text.png")
```
<img src="plots/geom_text.png?v=1" alt="geom_text" width="400px">

For `ggmulti` you can set where plots are empty using the 'empty_plots' variable. It can be either an int or list of ints. Note that the value is the index of the plot, starting from the top left and counting across. Also, as this is Python the indexes start at zero, so in a 2x3 grid, index no. 4 will be the bottom left had side.
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
    empty_plots=4
)
```
<img src="plots/gg_multi_pmg_with_one_empty.png?v=1" alt="gg_multi_pmg_with_one_empty" width="800px">

![gg](plots/simple_test.png?v=1)
