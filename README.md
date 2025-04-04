plots are still in progress, but some examples:

```python
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = ggplot(mpg, aes("class")) + geom_bar()
    res = ggcreate(plot)
    ggdraw_plot(res, data_path / "geom_bar.png")
```
<img src="data/geom_bar.png?v=1" alt="geom_bar" width="400px">

```python
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = ggplot(mpg, aes(x="displ", y="hwy")) + geom_point(
        aes(color="class"), size=3, alpha=0.7
    )

    res = ggcreate(plot)
    ggdraw_plot(res, data_path / "geom_point_with_color.png")
```
<img src="data/geom_point_with_color.png?v=1" alt="gg_point" width="400px">

```python
mpg = pd.read_csv(data_path / "mpg.csv")
plot = ggplot(mpg, aes(x = 'displ')) + geom_histogram()
res = ggcreate(plot)
ggdraw_plot(res, data_path / "geom_histogram.png")
```
<img src="data/geom_histogram.png?v=1" alt="geom_histogram" width="400px">

```python
    df = pd.DataFrame(
        data={"dose": ["D0.5", "D1", "D2"], "bbb": [4.2, 10, 29.5]}
    )
    plot = ggplot(df, aes(x="dose", y="bbb")) + geom_line() + geom_point()
    res = ggcreate(plot)
    ggdraw_plot(res, data_path / "geom_line_and_point_with_linetype.png")
```
<img src="data/geom_line_and_point_with_linetype.png?v=1" alt="geom_line_and_point" width="400px">

```python
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = ggplot(mpg, aes(x = 'displ', y = "cty", text = 'manufacturer')) + geom_text()
    res = ggcreate(plot)
    ggdraw_plot(res, data_path / "geom_text.png")
```
<img src="data/geom_text.png?v=1" alt="geom_text" width="400px">

![gg](data/simple_test.png?v=1)
