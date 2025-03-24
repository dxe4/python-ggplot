### Milestones :rock:
- :green_circle: Port ginger and ggplot logic :bar_chart:
    - :red_circle: clean up / refactor / fix todos :broom:
    - :red_circle: write tests :mag:
- :red_circle: Release alpha version :abc:
- :red_circle: Add additional features :building_construction:
    - :red_circle: features from R ggplot eg geom_hex
    - :red_circle: features skipped for alpha version eg TEX VEGA and FormulaNode
- :red_circle: Allow creating interactive plots :rocket:

plots are still in progress, but some examples:
```
    df = pd.DataFrame(
        data={"dose": ["D0.5", "D1", "D2"], "bbb": [4.2, 10, 29.5]}
    )
    plot = ggplot(df, aes(x="dose", y="bbb")) + geom_line() + geom_point()
    res = ggcreate(plot)
    ggdraw_plot(res, data_path / "geom_line_and_point.png")
```
![gg_line_and_point](data/geom_line_and_point.png?v=1)

```
    mpg = pd.read_csv(data_path / "mpg.csv")  # type: ignore
    plot = ggplot(mpg, aes("class")) + geom_bar()
    res = ggcreate(plot)
    ggdraw_plot(res, data_path / "geom_bar.png")
```
![gg_bar](data/geom_bar.png?v=1)
```
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = (
        ggplot(mpg, aes(x="displ", y="hwy", color="class"))
        + geom_point()
    )

    res = ggcreate(plot)
    ggdraw_plot(res, data_path / "geom_point.png")
```
![gg_point](data/geom_point.png?v=1)

```
    mpg = pd.read_csv(data_path / "mpg.csv")
    plot = ggplot(mpg, aes(x = 'displ', y = "cty", text = 'manufacturer')) + geom_text()
    res = ggcreate(plot)
    ggdraw_plot(res, data_path / "geom_text.png")
```
![gg_text](data/geom_text.png?v=1)

![gg](data/simple_test.png?v=1)
