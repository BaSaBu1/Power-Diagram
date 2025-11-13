import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from bowyer_watson import Point, Triangle, plot_triangulation

# --- Power Diagram (Weighted Delaunay) ---

def power_distance_sq(p1, p2):
    """
    Squared power distance between two weighted points
    power_dist^2 = euclidean_dist^2 - weight1^2
    """
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    euclidean_sq = dx*dx + dy*dy
    return euclidean_sq - p1.weight**2

class WeightedTriangle(Triangle):
    """Triangle with power-based circumcircle (orthocircle) calculation"""
    
    def circumcenter(self):
        """Calculate orthocircle for weighted sites"""
        p1, p2, p3 = self.vertices
        
        # unpacking points and weights
        ax, ay, w1 = p1.x, p1.y, p1.weight
        bx, by, w2 = p2.x, p2.y, p2.weight
        cx, cy, w3 = p3.x, p3.y, p3.weight
        
        # This is to bake in the weights into thepoints. To my understanding this is like making the point 3d and then projecting it back down to 2d.
        ax2 = ax*ax + ay*ay - w1
        bx2 = bx*bx + by*by - w2
        cx2 = cx*cx + cy*cy - w3
        
        # Use determinant to find area serve as a way to check for collinearity or if area is too small.
        D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(D) < 1e-10:
            return None
        
        # Determine the radical center by using the baked in weights. In a determinant and scale by area of triangle. Basically you can rewrite the radical line equation into the determinent and the same
        # and by summing up the orientations of each point to the other vertices axis values you get a dot product summing those and dividing gives you the intersection.
        ux = (ax2 * (by - cy) + bx2 * (cy - ay) + cx2 * (ay - by)) / D
        uy = (ax2 * (cx - bx) + bx2 * (ax - cx) + cx2 * (bx - ax)) / D
        

        # Relative to the Bowyer-Watson algo everything is the same except the initial power test with P = (0,0).
        return Point(ux, uy)
    
    def circumradius_sq(self):
        """Calculate squared power circle radius"""
        cc = self.circumcenter()
        if cc is None:
            return float('inf')
        
        # Power distance from center to first vertex this allows for comparision when checking other points around site.
        p = self.vertices[0]
        dx = p.x - cc.x
        dy = p.y - cc.y
        dist_sq = dx*dx + dy*dy
        # Adjust for weight
        return dist_sq - p.weight**2
    
    def contains_point_in_power_circle(self, p):
        """Check if weighted point p is inside the power circle"""
        cc = self.circumcenter()
        if cc is None:
            return False
        
        # Power distance from p to circumcenter
        dx = p.x - cc.x
        dy = p.y - cc.y
        dist_sq = dx*dx + dy*dy
        power_dist_sq = dist_sq - p.weight**2
        
        # Power circle radius
        rad_sq = self.circumradius_sq()
        
        return power_dist_sq < rad_sq - 1e-10

def weighted_delaunay(points):
    """
    Weighted Delaunay triangulation using power distances
    This creates a power diagram (Voronoi diagram for weighted sites)
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points")
    
    # Create super-triangle
    min_x = min(p.x for p in points)
    max_x = max(p.x for p in points)
    min_y = min(p.y for p in points)
    max_y = max(p.y for p in points)
    
    dx = max_x - min_x
    dy = max_y - min_y
    delta_max = max(dx, dy)
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2
    
    p1 = Point(mid_x - 20*delta_max, mid_y - delta_max, weight=0)
    p2 = Point(mid_x, mid_y + 20*delta_max, weight=0)
    p3 = Point(mid_x + 20*delta_max, mid_y - delta_max, weight=0)
    
    super_tri = WeightedTriangle(p1, p3, p2)
    triangles = [super_tri]
    
    # Insert points one by one
    for point in points:
        bad_triangles = []
        
        # Find all triangles whose power circle contains the point
        for tri in triangles:
            if tri.contains_point_in_power_circle(point):
                bad_triangles.append(tri)
        
        # Find boundary edges
        polygon = []
        for tri in bad_triangles:
            for i, edge_start in enumerate(tri.vertices):
                edge_end = tri.vertices[(i + 1) % 3]
                
                # Initialize flag
                shared = False
                # Check neighbors to see if they also share the same edge
                for other_tri in bad_triangles:
                    if other_tri == tri:
                        continue
                    for j, v in enumerate(other_tri.vertices):
                        # Using curr triangle(bad_triangle) as reference of points compare if they share same vertex orientation for shared edge case
                        if (v == edge_start and other_tri.vertices[(j + 1) % 3] == edge_end):
                            shared = True
                            break
                        if (v == edge_end and other_tri.vertices[(j + 1) % 3] == edge_start):
                            shared = True
                            break
                
                if not shared:
                    polygon.append((edge_start, edge_end))
        
        # Remove bad triangles
        for tri in bad_triangles:
            triangles.remove(tri)
        
        # Create new triangles
        for edge_start, edge_end in polygon:
            new_tri = WeightedTriangle(edge_start, edge_end, point)
            triangles.append(new_tri)
    
    # Remove triangles touching super-triangle vertices
    super_verts = {p1, p2, p3}
    final_triangles = [t for t in triangles 
                      if not any(v in super_verts for v in t.vertices)]
    
    return final_triangles

# Rendering helper function
def plot_power_diagram(triangles, points, title="Power Diagram"):
    """Plot Power Voronoi diagram"""
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot triangles
    for tri in triangles:
        verts = [(v.x, v.y) for v in tri.vertices]
        polygon = Polygon(verts, edgecolor='blue', facecolor='cyan', alpha=0.2, linewidth=1)
        ax.add_patch(polygon)
    
    # Plot power circles (orthocircles)
    '''
    for tri in triangles:
        cc = tri.circumcenter()
        if cc is not None:
            rad_sq = tri.circumradius_sq()
            if rad_sq > 0:
                rad = np.sqrt(rad_sq)
                circle = Circle((cc.x, cc.y), rad, edgecolor='green', facecolor='none', alpha=0.3, linewidth=0.5)
                ax.add_patch(circle)
    '''
    # Plot power Voronoi edges
    drawn_edges = set()
    for tri in triangles:
        cc = tri.circumcenter()
        if cc is None:
            continue
        
        for j in range(3):
            v1, v2 = tri.vertices[j], tri.vertices[(j + 1) % 3]
            # Find adjacent triangle
            for other_tri in triangles:
                if other_tri == tri:
                    continue
                for k in range(3):
                    ov1, ov2 = other_tri.vertices[k], other_tri.vertices[(k + 1) % 3]
                    if (v1 == ov1 and v2 == ov2) or (v1 == ov2 and v2 == ov1):
                        other_cc = other_tri.circumcenter()
                        if other_cc is not None:
                            edge_key = tuple(sorted([tri.id, other_tri.id]))
                            if edge_key not in drawn_edges:
                                ax.plot([cc.x, other_cc.x], [cc.y, other_cc.y], 'r-', linewidth=2)
                                drawn_edges.add(edge_key)
    
    # Plot input points with weights shown as circles
    for p in points:
        ax.plot(p.x, p.y, 'ko', markersize=8)
        if p.weight > 0:
            # Draw weight as a circle
            circle = Circle((p.x, p.y), p.weight, edgecolor='black', facecolor='none', 
                          linestyle='--', linewidth=1.5, alpha=0.6, label='weight' if p == points[0] else '')
            ax.add_patch(circle)
    
    ax.set_aspect('equal')
    ax.set_title(title)
    
    all_x = [p.x for p in points]
    all_y = [p.y for p in points]
    ax.set_xlim(min(all_x)-2, max(all_x)+2)
    ax.set_ylim(min(all_y)-2, max(all_y)+2)
    
    return fig, ax


def plot_power_voronoi_shaded(triangles, points, title="Power Voronoi (shaded)"):
    """Shade power-Voronoi regions by computing each site's cell via half-plane clipping.

    This uses the power bisector half-planes between sites and clips to a plotting
    bounding box so unbounded cells become finite polygons suitable for shading.
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    # Bounding box (slightly padded)
    all_x = [p.x for p in points]
    all_y = [p.y for p in points]
    xmin, xmax = min(all_x) - 2, max(all_x) + 2
    ymin, ymax = min(all_y) - 2, max(all_y) + 2
    # initial clipping polygon (rectangle CCW)
    bbox = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

    def clip_polygon_with_halfplane(poly, A, B, C):
        """Clip polygon (list of (x,y)) with half-plane A*x + B*y <= C.
        Uses Sutherland-Hodgman polygon clipping."""
        if not poly:
            return []
        out = []
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1) % n]
            v1 = A*x1 + B*y1 - C <= 1e-9
            v2 = A*x2 + B*y2 - C <= 1e-9
            if v1 and v2:
                out.append((x2, y2))
            elif v1 and not v2:
                # leaving
                denom = A*(x2 - x1) + B*(y2 - y1)
                if abs(denom) > 1e-12:
                    t = (C - A*x1 - B*y1) / denom
                    xi = x1 + t*(x2 - x1)
                    yi = y1 + t*(y2 - y1)
                    out.append((xi, yi))
            elif not v1 and v2:
                # entering
                denom = A*(x2 - x1) + B*(y2 - y1)
                if abs(denom) > 1e-12:
                    t = (C - A*x1 - B*y1) / denom
                    xi = x1 + t*(x2 - x1)
                    yi = y1 + t*(y2 - y1)
                    out.append((xi, yi))
                out.append((x2, y2))
            # else both outside -> nothing
        return out

    cmap = plt.get_cmap('tab10')
    for i, pi in enumerate(points):
        # start from bbox and intersect half-planes: power(pi,x) <= power(pj,x)
        poly = bbox[:]
        pi_val = pi.x*pi.x + pi.y*pi.y - pi.weight**2
        for j, pj in enumerate(points):
            if i == j:
                continue
            pj_val = pj.x*pj.x + pj.y*pj.y - pj.weight**2
            # inequality: 2*(pj - pi) . x <= pj_val - pi_val
            A = 2*(pj.x - pi.x)
            B = 2*(pj.y - pi.y)
            C = pj_val - pi_val
            poly = clip_polygon_with_halfplane(poly, A, B, C)
            if not poly:
                break
        if poly:
            xs, ys = zip(*poly)
            ax.fill(xs, ys, color=cmap(i % 10), alpha=0.25)

    # Plot sites and labels
    for i, p in enumerate(points):
        ax.plot(p.x, p.y, 'ko')
        ax.text(p.x + 0.08, p.y + 0.08, str(i+1), fontsize=10)

    ax.set_aspect('equal')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    return fig, ax


def shade_power_voronoi_on_ax(points, ax, padding=2.0, alpha=0.25, cmap_name='tab10'):
    """Shade power-Voronoi regions for `points` onto the provided Axes `ax`.

    This reuses the same half-plane clipping logic as `plot_power_voronoi_shaded`
    but draws into an existing axes so it can be composed into multi-panel
    figures (e.g., a left/right comparison).
    """
    # Bounding box
    all_x = [p.x for p in points]
    all_y = [p.y for p in points]
    xmin, xmax = min(all_x) - padding, max(all_x) + padding
    ymin, ymax = min(all_y) - padding, max(all_y) + padding
    bbox = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

    def clip_polygon_with_halfplane(poly, A, B, C):
        if not poly:
            return []
        out = []
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1) % n]
            v1 = A*x1 + B*y1 - C <= 1e-9
            v2 = A*x2 + B*y2 - C <= 1e-9
            if v1 and v2:
                out.append((x2, y2))
            elif v1 and not v2:
                denom = A*(x2 - x1) + B*(y2 - y1)
                if abs(denom) > 1e-12:
                    t = (C - A*x1 - B*y1) / denom
                    xi = x1 + t*(x2 - x1)
                    yi = y1 + t*(y2 - y1)
                    out.append((xi, yi))
            elif not v1 and v2:
                denom = A*(x2 - x1) + B*(y2 - y1)
                if abs(denom) > 1e-12:
                    t = (C - A*x1 - B*y1) / denom
                    xi = x1 + t*(x2 - x1)
                    yi = y1 + t*(y2 - y1)
                    out.append((xi, yi))
                out.append((x2, y2))
        return out

    cmap = plt.get_cmap(cmap_name)
    for i, pi in enumerate(points):
        poly = bbox[:]
        pi_val = pi.x*pi.x + pi.y*pi.y - pi.weight**2
        for j, pj in enumerate(points):
            if i == j:
                continue
            pj_val = pj.x*pj.x + pj.y*pj.y - pj.weight**2
            A = 2*(pj.x - pi.x)
            B = 2*(pj.y - pi.y)
            C = pj_val - pi_val
            poly = clip_polygon_with_halfplane(poly, A, B, C)
            if not poly:
                break
        if poly:
            xs, ys = zip(*poly)
            ax.fill(xs, ys, color=cmap(i % 10), alpha=alpha)

    # plot sites (markers and labels left to caller)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')


def generate_random_points(n=8, x_range=(0,10), y_range=(0,10), weight_range=(0.0, 3.0), seed=None):
    """Generate n random simple Point-like objects with one-decimal coords and weights.

    Returns objects with .x, .y, .weight attributes (compatible with shading helper).
    If seed is provided the RNG is seeded for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)
    xs = np.round(np.random.uniform(x_range[0], x_range[1], size=(n,)), 1)
    ys = np.round(np.random.uniform(y_range[0], y_range[1], size=(n,)), 1)
    ws = np.round(np.random.uniform(weight_range[0], weight_range[1], size=(n,)), 2)

    class SimplePoint:
        def __init__(self, x, y, weight=0.0):
            self.x = float(x)
            self.y = float(y)
            self.weight = float(weight)

    return [SimplePoint(float(x), float(y), float(w)) for x, y, w in zip(xs, ys, ws)]

# --- Main ---
if __name__ == "__main__":
    Point._id_counter = 0
    Triangle._id_counter = 0
    
    # Test with weighted points
    weighted_data = [
        (1, 1, 10),
        (9, 1, 0),
        (9, 9, 20),
        (1, 9, 50),
        (5, 2, 28),
        (3, 7, 5),
        (7, 7, 9),
        (5, 5, 2)
    ]
    
    points = [Point(x, y, w) for x, y, w in weighted_data]
    
    print(f"Computing Weighted Delaunay (Power Diagram) for {len(points)} weighted points...")
    triangles = weighted_delaunay(points)
    
    print(f"Result: {len(triangles)} triangles")
    
    # Check points
    print("\nWeighted points in triangulation:")
    for p in points:
        found = False
        for tri in triangles:
            if p in tri.vertices:
                found = True
                break
        status = "✓" if found else "✗"
        print(f"  {status} ({p.x:.1f}, {p.y:.1f}) weight={p.weight:.1f}")
    
    
    print("\nPlot saved to power_diagram.png")

    # Show shaded power-Voronoi (vector polygons)
    fig2, ax2 = plot_power_voronoi_shaded(triangles, points, "Power Voronoi (shaded)")
    plt.show()
