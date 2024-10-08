import matplotlib.pyplot as plt 
import numpy as np

from src.objects import *

#TODO: add cases for Triangle and DoubleRectangle in draw_grid(...)

def draw_grid(ax,scene_obj):
    if isinstance(scene_obj,Rectangle):

        # Draw the grid
        cell_size = scene_obj.cell_size  # Size of each grid cell

        # Iterate over the grid and draw cells
        for i in range(scene_obj.grid_width):
            for j in range(scene_obj.grid_height):
                # Determine the cell boundaries
                x_start = i * cell_size
                y_start = j * cell_size
                x_end = (i + 1) * cell_size
                y_end = (j + 1) * cell_size

                # Check if this is an edge cell
                if scene_obj.grid[i][j][2]:  # Edge cell
                    rect = plt.Rectangle((x_start, y_start), cell_size, cell_size, 
                                        linewidth=1, edgecolor='red', linestyle='-', fill=False)
                else:  # Non-edge cell
                    rect = plt.Rectangle((x_start, y_start), cell_size, cell_size, 
                                        linewidth=1, edgecolor='black', linestyle='-', fill=False)

                # Add the rectangle (cell) to the plot
                ax.add_patch(rect)


    elif isinstance(scene_obj,Circle):
        # Draw the grid
        cell_size = scene_obj.cell_size  # Size of each grid cell
        grid_center_x = scene_obj.grid_width / 2
        grid_center_y = scene_obj.grid_height / 2

        # Iterate over the grid and draw cells, centered on the circle
        for i in range(scene_obj.grid_width):
            for j in range(scene_obj.grid_height):
                # Determine the cell's center relative to the origin (0, 0)
                cell_center_x = (i + 0.5 - grid_center_x) * cell_size
                cell_center_y = (j + 0.5 - grid_center_y) * cell_size

                # Determine the cell boundaries (based on the center)
                x_start = cell_center_x - cell_size / 2
                y_start = cell_center_y - cell_size / 2
                x_end = cell_center_x + cell_size / 2
                y_end = cell_center_y + cell_size / 2

                # Check if this is an edge cell
                if scene_obj.grid[i][j][2]:  # Edge cell
                    rect = plt.Rectangle((x_start, y_start), cell_size, cell_size, 
                                        linewidth=1, edgecolor='red', linestyle='-', fill=False)
                else:  # Non-edge cell
                    rect = plt.Rectangle((x_start, y_start), cell_size, cell_size, 
                                        linewidth=1, edgecolor='black', linestyle='-', fill=False)

                # Add the rectangle (cell) to the plot
                ax.add_patch(rect)