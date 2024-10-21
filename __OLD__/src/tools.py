import numpy as np
import random
import math
from collections import deque
from typing import List, Tuple


def poisson_disk_sampling_strict_no_overlap(N: int, radii: List[float], shape: Tuple, k: int = 30) -> List[Tuple[float, float, float]]:
    """
    Poisson disk sampling to generate non-overlapping circles of varying radii within a given shape.
    Uses spatial hashing to optimize overlap checks and prevent overlapping of circles.

    :param N: The number of circles (particles) to generate.
    :param radii: List of radii for the circles. Radii are selected randomly from this list for each circle.
    :param shape: A tuple representing the shape in which to generate circles. The following shapes are supported:
    
        - 'rectangle': ('rectangle', (x, y, width, height))
            - x: X-coordinate of the bottom-left corner of the rectangle.
            - y: Y-coordinate of the bottom-left corner of the rectangle.
            - width: Width of the rectangle.
            - height: Height of the rectangle.
        
        - 'circle': ('circle', (center_x, center_y, radius))
            - center_x: X-coordinate of the center of the enclosing circle.
            - center_y: Y-coordinate of the center of the enclosing circle.
            - radius: Radius of the enclosing circle.
        
        - 'triangle': ('triangle', (p1, p2, p3))
            - p1: Coordinates of lower left vertex of the triangle as (x1, y1).
            - p2: Coordinates of lower right vertex of the triangle as (x2, y2).
            - p3: Coordinates of top vertex of the triangle as (x3, y3).
        
        - 'polygon': ('polygon', [(x1, y1), (x2, y2), ..., (xn, yn)])
            - A list of vertices (x, y) defining the polygon. The polygon can have any number of vertices.
    
    :param k: Number of candidate points to try for each point before giving up (default is 30).
    :return: A list of circles as tuples (x, y, radius), where:
        - x: X-coordinate of the circle center.
        - y: Y-coordinate of the circle center.
        - radius: Radius of the circle.
    """
    
    shape_type = shape[0]
    
    # Get the bounding box or the shape boundaries for grid calculation
    if shape_type == 'rectangle':
        rect_x, rect_y, rect_width, rect_height = shape[1]
    elif shape_type in ['triangle', 'polygon', 'circle']:
        rect_x, rect_y, rect_width, rect_height = get_shape_bounding_box(shape)
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")
    
    # Calculate grid cell size based on the largest radius
    r_max = max(radii)
    cell_size = r_max / math.sqrt(2)
    
    # Determine the dimensions of the grid
    grid_width = int(rect_width / cell_size) + 1
    grid_height = int(rect_height / cell_size) + 1
    grid = [[None for _ in range(grid_height)] for _ in range(grid_width)]
    
    # Active list of points (the current set of circles being processed)
    active_list = deque()
    
    # List of all circles generated
    packed_circles = []
    
    def add_point(x: float, y: float, radius: float) -> None:
        """
        Adds a circle to the spatial grid and active list.

        :param x: X-coordinate of the circle center.
        :param y: Y-coordinate of the circle center.
        :param radius: Radius of the circle.
        """
        gx = int((x - rect_x) / cell_size)
        gy = int((y - rect_y) / cell_size)
        if grid[gx][gy] is None:
            grid[gx][gy] = []
        grid[gx][gy].append((x, y, radius))
        packed_circles.append((x, y, radius))
        active_list.append((x, y, radius))

    def check_no_overlap(new_x: float, new_y: float, new_radius: float) -> bool:
        """
        Checks if a circle overlaps with any other circle in the nearby grid cells.

        :param new_x: X-coordinate of the circle center.
        :param new_y: Y-coordinate of the circle center.
        :param new_radius: Radius of the new circle.
        :return: True if no overlap, False otherwise.
        """
        gx = int((new_x - rect_x) / cell_size)
        gy = int((new_y - rect_y) / cell_size)
        
        search_radius = int((2 * new_radius) / cell_size) + 1
        
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                nx = gx + dx
                ny = gy + dy
                if 0 <= nx < grid_width and 0 <= ny < grid_height:
                    if grid[nx][ny] is not None:
                        for (px, py, pr) in grid[nx][ny]:
                            if distance((new_x, new_y), (px, py)) < (new_radius + pr):
                                return False
        return True
    
    # Start by placing a circle at the center of the shape's bounding box
    first_radius = radii[0]
    first_point = (rect_x + rect_width / 2, rect_y + rect_height / 2)
    
    # Ensure that the first point is inside the shape
    if is_point_in_shape(first_point[0], first_point[1], first_radius, shape):
        add_point(first_point[0], first_point[1], first_radius)
    
    while active_list and len(packed_circles) < N:
        # Randomly select a point from the active list
        i = random.randint(0, len(active_list) - 1)
        x, y, radius = active_list[i]
        
        # Try to generate k points around the current circle
        for _ in range(k):
            new_radius = random.choice(radii)
            r = random.uniform(radius + new_radius, 2 * (radius + new_radius))
            theta = random.uniform(0, 2 * math.pi)
            new_x = x + r * math.cos(theta)
            new_y = y + r * math.sin(theta)
            
            # Check if the new point is inside the shape and does not overlap
            if is_point_in_shape(new_x, new_y, new_radius, shape):
                if check_no_overlap(new_x, new_y, new_radius):
                    add_point(new_x, new_y, new_radius)
                    if len(packed_circles) >= N:
                        break
        
        # If no valid point is found after k tries, remove it from the active list
        if len(packed_circles) >= N:
            break
        else:
            active_list.remove((x, y, radius))
    
    return packed_circles


def is_point_in_shape(x: float, y: float, radius: float, shape: Tuple) -> bool:
    """
    Check if a circle with the given radius is inside a shape.
    
    :param x: X-coordinate of the circle center.
    :param y: Y-coordinate of the circle center.
    :param radius: Radius of the circle.
    :param shape: A tuple representing the shape in which to generate circles. The following shapes are supported:
    
        - 'rectangle': ('rectangle', (x, y, width, height))
            - x: X-coordinate of the bottom-left corner of the rectangle.
            - y: Y-coordinate of the bottom-left corner of the rectangle.
            - width: Width of the rectangle.
            - height: Height of the rectangle.
        
        - 'circle': ('circle', (center_x, center_y, radius))
            - center_x: X-coordinate of the center of the enclosing circle.
            - center_y: Y-coordinate of the center of the enclosing circle.
            - radius: Radius of the enclosing circle.
        
        - 'triangle': ('triangle', (p1, p2, p3))
            - p1: Coordinates of lower left vertex of the triangle as (x1, y1).
            - p2: Coordinates of lower right vertex of the triangle as (x2, y2).
            - p3: Coordinates of top vertex of the triangle as (x3, y3).
        
        - 'polygon': ('polygon', [(x1, y1), (x2, y2), ..., (xn, yn)])
            - A list of vertices (x, y) defining the polygon. The polygon can have any number of vertices.
    :return: True if the circle is fully inside the shape, False otherwise.
    """
    shape_type = shape[0]
    
    if shape_type == 'rectangle':
        return is_circle_in_rectangle(x, y, radius, shape[1])
    
    elif shape_type == 'triangle':
        return is_circle_in_triangle(x,y,radius,shape[1])
    
    elif shape_type == 'polygon':
        #TODO: Placeholder for future implementation
        pass
    
    elif shape_type == 'circle':
        return is_circle_in_circle(x, y, radius, shape[1])
    
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")


def is_circle_in_rectangle(x: float, y: float, radius: float, rectangle: Tuple[float, float, float, float]) -> bool:
    """
    Check if a circle is fully inside a rectangle.
    
    :param x: X-coordinate of the circle center.
    :param y: Y-coordinate of the circle center.
    :param radius: Radius of the circle.
    :param rectangle: Tuple representing the rectangle (x, y, width, height).
    :return: True if the circle is fully inside the rectangle, False otherwise.
    """
    rect_x, rect_y, width, height = rectangle
    return rect_x + radius <= x <= rect_x + width - radius and rect_y + radius <= y <= rect_y + height - radius


def is_circle_in_circle(x: float, y: float, radius: float, enclosing_circle: Tuple[float, float, float]) -> bool:
    """
    Check if a circle is fully inside another circle.
    
    :param x: X-coordinate of the inner circle's center.
    :param y: Y-coordinate of the inner circle's center.
    :param radius: Radius of the inner circle.
    :param enclosing_circle: Tuple representing the enclosing circle (center_x, center_y, radius).
    :return: True if the inner circle is fully inside the enclosing circle, False otherwise.
    """
    center_x, center_y, enclosing_radius = enclosing_circle
    dist = distance((x, y), (center_x, center_y))
    return dist + radius <= enclosing_radius

def is_circle_in_triangle(x: float, y: float, radius: float, enclosing_triangle: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]) -> bool:
    """
    Check if a circle is fully inside a triangle.

    :param x: X-coordinate of the circle's center.
    :param y: Y-coordinate of the circle's center.
    :param radius: Radius of the circle.
    :param enclosing_triangle: Tuple representing the triangle's vertices (p1, p2, p3).
    :return: True if the circle is fully inside the triangle, False otherwise.
    """
    def point_to_line_distance(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        """
        Calculate the distance from a point to a line segment.
        """
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        t = np.dot(line_unitvec, point_vec / line_len)
        t = max(0.0, min(1.0, t))  # Project onto the segment
        nearest = line_start + line_unitvec * t * line_len
        dist = np.linalg.norm(point - nearest)
        return dist

    point = np.array([x, y])
    v1 = np.array(enclosing_triangle[0])
    v2 = np.array(enclosing_triangle[1])
    v3 = np.array(enclosing_triangle[2])

    # First, check if the circle's center is inside the triangle
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(point, v1, v2)
    d2 = sign(point, v2, v3)
    d3 = sign(point, v3, v1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    is_center_inside = not (has_neg and has_pos)

    # Check if the circle's center is inside
    if is_center_inside:
        # Now, ensure the circle's radius doesn't overlap with the triangle's edges
        for edge in [(v1, v2), (v2, v3), (v3, v1)]:
            if point_to_line_distance(point, edge[0], edge[1]) < radius:
                return False  # Circle overlaps with an edge

        return True

    return False




def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two points.

    :param p1: First point as (x, y).
    :param p2: Second point as (x, y).
    :return: Euclidean distance between the two points.
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_shape_bounding_box(shape: Tuple) -> Tuple[float, float, float, float]:
    """
    Calculate the bounding box of a given shape.
    
    :param shape: A tuple representing the shape.
    :return: A tuple representing the bounding box as (x, y, width, height).
    """
    shape_type = shape[0]
    
    if shape_type == 'rectangle':
        return shape[1]  # Already a bounding box
    
    elif shape_type == 'triangle':
        p1, p2, p3 = shape[1]
        min_x = min(p1[0], p2[0], p3[0])
        max_x = max(p1[0], p2[0], p3[0])
        min_y = min(p1[1], p2[1], p3[1])
        max_y = max(p1[1], p2[1], p3[1])
        return min_x, min_y, max_x - min_x, max_y - min_y
    
    elif shape_type == 'polygon':
        vertices = shape[1]
        min_x = min(v[0] for v in vertices)
        max_x = max(v[0] for v in vertices)
        min_y = min(v[1] for v in vertices)
        max_y = max(v[1] for v in vertices)
        return min_x, min_y, max_x - min_x, max_y - min_y
    
    elif shape_type == 'circle':
        center_x, center_y, radius = shape[1]
        return center_x - radius, center_y - radius, 2 * radius, 2 * radius
    
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")



def line_intersects_rectangle(rect_bottom_left_x: float, rect_bottom_left_y: float, rect_top_right_x: float, rect_top_right_y: float, 
                              line_start_x: float, line_start_y: float, line_end_x: float, line_end_y: float) -> bool:
    """
    Check if a line segment intersects with a rectangle using the Axis-Aligned Bounding Box (AABB) method.

    This function uses the Cohen-Sutherland line clipping algorithm to determine if the line segment 
    defined by the endpoints (line_start_x, line_start_y) and (line_end_x, line_end_y) intersects with the rectangle defined by its 
    bottom-left corner (rect_bottom_left_x, rect_bottom_left_y) and top-right corner (rect_top_right_x, rect_top_right_y).

    Args:
        rect_bottom_left_x (float): X-coordinate of the bottom-left corner of the rectangle.
        rect_bottom_left_y (float): Y-coordinate of the bottom-left corner of the rectangle.
        rect_top_right_x (float): X-coordinate of the top-right corner of the rectangle.
        rect_top_right_y (float): Y-coordinate of the top-right corner of the rectangle.
        line_start_x (float): X-coordinate of the starting point of the line segment.
        line_start_y (float): Y-coordinate of the starting point of the line segment.
        line_end_x (float): X-coordinate of the ending point of the line segment.
        line_end_y (float): Y-coordinate of the ending point of the line segment.

    Returns:
        bool: True if the line segment intersects the rectangle, False otherwise.
    """
    # Ensure the rectangle coordinates are ordered correctly
    rect_bottom_left_x, rect_top_right_x = min(rect_bottom_left_x, rect_top_right_x), max(rect_bottom_left_x, rect_top_right_x)
    rect_bottom_left_y, rect_top_right_y = min(rect_bottom_left_y, rect_top_right_y), max(rect_bottom_left_y, rect_top_right_y)

    # Check if line segment is completely outside the rectangle's bounding box
    if max(line_start_x, line_end_x) < rect_bottom_left_x or min(line_start_x, line_end_x) > rect_top_right_x or max(line_start_y, line_end_y) < rect_bottom_left_y or min(line_start_y, line_end_y) > rect_top_right_y:
        return False

    # Check for intersection by using Cohen-Sutherland line clipping algorithm
    # Define region codes for each point of the line segment
    INSIDE = 0  # 0000
    LEFT = 1    # 0001
    RIGHT = 2   # 0010
    BOTTOM = 4  # 0100
    TOP = 8     # 1000

    def compute_code(x: float, y: float) -> int:
        code = INSIDE
        if x < rect_bottom_left_x:
            code |= LEFT
        elif x > rect_top_right_x:
            code |= RIGHT
        if y < rect_bottom_left_y:
            code |= BOTTOM
        elif y > rect_top_right_y:
            code |= TOP
        return code

    code1 = compute_code(line_start_x, line_start_y)
    code2 = compute_code(line_end_x, line_end_y)

    while True:
        if code1 == 0 and code2 == 0:
            return True  # Trivially inside
        elif code1 & code2 != 0:
            return False  # Trivially outside
        else:
            # Clip the line to the rectangle's bounding box
            code_out = code1 if code1 != 0 else code2
            if code_out & TOP:
                x = line_start_x + (line_end_x - line_start_x) * (rect_top_right_y - line_start_y) / (line_end_y - line_start_y)
                y = rect_top_right_y
            elif code_out & BOTTOM:
                x = line_start_x + (line_end_x - line_start_x) * (rect_bottom_left_y - line_start_y) / (line_end_y - line_start_y)
                y = rect_bottom_left_y
            elif code_out & RIGHT:
                y = line_start_y + (line_end_y - line_start_y) * (rect_top_right_x - line_start_x) / (line_end_x - line_start_x)
                x = rect_top_right_x
            elif code_out & LEFT:
                y = line_start_y + (line_end_y - line_start_y) * (rect_bottom_left_x - line_start_x) / (line_end_x - line_start_x)
                x = rect_bottom_left_x

            if code_out == code1:
                line_start_x, line_start_y = x, y
                code1 = compute_code(line_start_x, line_start_y)
            else:
                line_end_x, line_end_y = x, y
                code2 = compute_code(line_end_x, line_end_y)