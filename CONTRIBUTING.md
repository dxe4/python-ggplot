This is a draft version to help people who want to start on the project, and it will evolve over time.
It has 2 sections

1) A few notes on contributing, current status of the package
2) A quick overview of the most important components starting from low level

## Notes:
We need tests and documentation on using the library.
good examples of what to test will include examples from here
https://r-graph-gallery.com/
https://r-statistics.co/Top50-Ggplot2-Visualizations-MasterList-R-Code.html

Some functionalitity will be missing,
so not all those plots can be implement from the packages as is without implementing for example some geoms.
but lot of the functionality can be done already

There are a few github issues documenting some bugs and work to be done.

We also need many unit tests, but testing specific plots maybe an easier start

## Quick overview of the implementation

[Cairo Backend](src/python_ggplot/graphics/cairo_backend.py)
Eventually all code ends up here, which draws the PNG
There is a plan for multiple backends in the future
It's unlikely you will need to change this,
unless you are making a new geom that needs custom functions

[Coords](src/python_ggplot/core/coord/objects.py)
[Quantities](src/python_ggplot/core/units/objects.py)
[UnitType](src/python_ggplot/core/objects.py)
Coords/Quantities is the base for the coordinate system and length measuring.
They both use UnitType.
Often you deal with length units (POINT,CENTIMETER,INCH) or relative (RELATIVE)
Relative of 0.8 means 80%
Conversion utilities exist, to convert from one type to another, add multiply etc

[GraphicsObjects](src/python_ggplot/graphics/objects.py)
This consists of graphics objects to be drawn by the cairo backend.
for example geom_bar will become GORect,
which eventually gets drawn by cairo backend draw_rectangle

[ViewPort](src/python_ggplot/graphics/views.py)
The image is a viewport with some height and width and graphics objects to render
the viewport also has sub view ports
the functions draw_viewport and transform_and_draw in [draw.py](src/python_ggplot/graphics/draw.py)
embed the graphcis objects in the views and draw them

### ggplot objects
for the time being, from here and on for anything unfamiliar ggplot should be used to familiarise with it.
eventually this part will be more documented.

[GgPlot](src/python_ggplot/gg/types.py)
holds all high level objects: df, aes, theme, geoms, facet, ridges

[Geom](src/python_ggplot/gg/geom/base.py)
low level geom implementaltion

[Scales](src/python_ggplot/gg/scales/base.py)
low level scale implementation

[public_interface](src/python_ggplot/public_interface)
This contains everything a user will need (but for now it also contains some extra logic)
This needs some cleaning up

Example:
```
from python_ggplot.public_interface.aes import aes
from python_ggplot.public_interface.common import ggdraw_plot, ggtitle
from python_ggplot.public_interface.geom import geom_bar, geom_point, ggplot
from python_ggplot.public_interface.utils import ggcreate
```
