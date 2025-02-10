from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Slider
import numpy as np
from bokeh.layouts import column, gridplot

# Initial parameter
alpha = 1.0
x = np.linspace(0, alpha, 100)

# Create data sources for multiple lines
source_sin = ColumnDataSource(data={'x': x, 'y1': np.sin(2 * np.pi * x), 'y2': np.sin(4 * np.pi * x)})
source_cos = ColumnDataSource(data={'x': x, 'y1': np.cos(2 * np.pi * x), 'y2': np.cos(4 * np.pi * x)})
source_tan = ColumnDataSource(data={'x': x, 'y1': np.tan(2 * np.pi * x), 'y2': np.tan(4 * np.pi * x)})
source_exp = ColumnDataSource(data={'x': x, 'y1': np.exp(-x), 'y2': np.exp(-2*x)})

# Create figures with multiple lines
p1 = figure(title="Sine Waves", width=400, height=300, x_axis_label="x", y_axis_label="y", x_range=(0, alpha), y_range=(-1.1, 1.1))
p1.line('x', 'y1', source=source_sin, line_width=2, color="blue", legend_label="sin(2πx)")
p1.line('x', 'y2', source=source_sin, line_width=2, color="red", legend_label="sin(4πx)")
p1.legend.location = "top_right"

p2 = figure(title="Cosine Waves", width=400, height=300, x_axis_label="x", y_axis_label="y", x_range=(0, alpha), y_range=(-1.1, 1.1))
p2.line('x', 'y1', source=source_cos, line_width=2, color="green", legend_label="cos(2πx)")
p2.line('x', 'y2', source=source_cos, line_width=2, color="purple", legend_label="cos(4πx)")
p2.legend.location = "top_right"

p3 = figure(title="Tangent Waves (Limited y-axis)", width=400, height=300, x_axis_label="x", y_axis_label="y", x_range=(0, alpha), y_range=(-5, 5))
p3.line('x', 'y1', source=source_tan, line_width=2, color="orange", legend_label="tan(2πx)")
p3.line('x', 'y2', source=source_tan, line_width=2, color="brown", legend_label="tan(4πx)")
p3.legend.location = "top_right"

p4 = figure(title="Exponential Decay", width=400, height=300, x_axis_label="x", y_axis_label="y", x_range=(0, alpha), y_range=(0, 1.1))
p4.line('x', 'y1', source=source_exp, line_width=2, color="black", legend_label="e^(-x)")
p4.line('x', 'y2', source=source_exp, line_width=2, color="gray", legend_label="e^(-2x)")
p4.legend.location = "top_right"

# Create slider
slider = Slider(start=0.1, end=2.0, value=alpha, step=0.1, title="Alpha (Range of x)")

# Update function
def update(attr, old, new):
    alpha_new = slider.value
    x_new = np.linspace(0, alpha_new, 100)

    # Update sources
    source_sin.data = {'x': x_new, 'y1': np.sin(2 * np.pi * x_new), 'y2': np.sin(4 * np.pi * x_new)}
    source_cos.data = {'x': x_new, 'y1': np.cos(2 * np.pi * x_new), 'y2': np.cos(4 * np.pi * x_new)}
    source_tan.data = {'x': x_new, 'y1': np.tan(2 * np.pi * x_new), 'y2': np.tan(4 * np.pi * x_new)}
    source_exp.data = {'x': x_new, 'y1': np.exp(-x_new), 'y2': np.exp(-2*x_new)}

    # Dynamically update x-axis range
    for plot in [p1, p2, p3, p4]:
        plot.x_range.start = 0
        plot.x_range.end = alpha_new

slider.on_change('value', update)

# Arrange layout
layout = column(slider, gridplot([[p1, p2], [p3, p4]]))
curdoc().add_root(layout)


