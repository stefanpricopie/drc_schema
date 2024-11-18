import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_contour(ax, obj):
    # Create a grid of points in 2D space
    x = np.linspace(*obj.bounds.T[0], 100)
    y = np.linspace(*obj.bounds.T[1], 100)
    X_grid, Y_grid = np.meshgrid(x, y)

    # Evaluate the function on the grid
    points = torch.tensor(np.c_[X_grid.ravel(), Y_grid.ravel()], dtype=torch.float32)
    Z = obj(points).view(100, 100).detach().numpy()

    # Plot the contour on the provided axis
    contour = ax.contour(X_grid, Y_grid, Z, cmap= mpl.colormaps['viridis_r'])

    return ax


def plot_with_filled_rectangle(ax, x_interval, y_interval, fill_color='green', alpha=0.3):
    # Define the x and y intervals for the rectangle
    x_start = x_interval[0]  # Start of x interval
    x_width = x_interval[1] - x_interval[0]  # Width of the rectangle
    y_start = y_interval[0]  # Start of y interval
    y_height = y_interval[1] - y_interval[0]  # Height of the rectangle

    # Add a filled rectangle to the plot
    rectangle = plt.Rectangle((x_start, y_start), x_width, y_height,
                              linewidth=1, edgecolor=fill_color, facecolor=fill_color, alpha=alpha, label='Constraint region')

    # Add the rectangle to the axis
    ax.add_patch(rectangle)

    return ax
