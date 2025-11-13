import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# --- Data Structures ---
class Point:
    _id_counter = 0
    def __init__(self, x, y, weight=0):
        self.x = x
        self.y = y
        self.weight = weight  # For weighted/power diagrams
        self.id = Point._id_counter
        Point._id_counter += 1
    
    def __repr__(self):
        return f"P({self.x:.2f}, {self.y:.2f})"
    
    def __eq__(self, other):
        return self.id == other.id
    
    def __hash__(self):
        return hash(self.id)

class Triangle:
    _id_counter = 0
    def __init__(self, v1, v2, v3):
        self.vertices = [v1, v2, v3]
        self.id = Triangle._id_counter
        Triangle._id_counter += 1
    
    def __repr__(self):
        return f"T{self.id}"
    
    def circumcenter(self):
        """Calculate circumcenter of triangle"""
        ax, ay = self.vertices[0].x, self.vertices[0].y
        bx, by = self.vertices[1].x, self.vertices[1].y
        cx, cy = self.vertices[2].x, self.vertices[2].y
        
        # find the area of the triangle via cross product
        D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(D) < 1e-10:
            return None
        # so this is just standard way to find circumcenter
        ux = ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / D
        uy = ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / D
        return Point(ux, uy)
    
    def circumradius_sq(self):
        """Calculate squared circumradius"""
        cc = self.circumcenter()
        if cc is None:
            return float('inf')
        dx = self.vertices[0].x - cc.x
        dy = self.vertices[0].y - cc.y
        return dx*dx + dy*dy
    
    def contains_point_in_circumcircle(self, p):
        """Check if point p is strictly inside circumcircle"""
        cc = self.circumcenter()
        if cc is None:
            return False
        
        dist_sq = (p.x - cc.x)**2 + (p.y - cc.y)**2
        rad_sq = self.circumradius_sq()
        return dist_sq < rad_sq - 1e-10  # Small epsilon for numerical stability

# --- Core Algorithm ---
def bowyer_watson(points):
    """
    Standard Bowyer-Watson incremental Delaunay triangulation
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points")
    
    # Create super-triangle - large enough to contain all points
    min_x = min(p.x for p in points)
    max_x = max(p.x for p in points)
    min_y = min(p.y for p in points)
    max_y = max(p.y for p in points)
    
    dx = max_x - min_x
    dy = max_y - min_y
    delta_max = max(dx, dy)
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2
    
    # Super-triangle with large coordinates
    p1 = Point(mid_x - 20*delta_max, mid_y - delta_max)
    p2 = Point(mid_x, mid_y + 20*delta_max)
    p3 = Point(mid_x + 20*delta_max, mid_y - delta_max)
    
    super_tri = Triangle(p1, p3, p2)  # CCW winding
    triangles = [super_tri]
    
    # Insert points one by one
    for point in points:
        bad_triangles = []
        
        # Find all triangles whose circumcircle contains the point
        for tri in triangles:
            if tri.contains_point_in_circumcircle(point):
                bad_triangles.append(tri)
        
        # Find the boundary of the polygonal hole
        polygon = []
        for tri in bad_triangles:
            for i, edge_start in enumerate(tri.vertices):
                edge_end = tri.vertices[(i + 1) % 3]
                
                # Check if edge is shared with another bad triangle
                shared = False
                for other_tri in bad_triangles:
                    if other_tri == tri:
                        continue
                    # Check if edge is in other_tri
                    for j, v in enumerate(other_tri.vertices):
                        if (v == edge_start and other_tri.vertices[(j + 1) % 3] == edge_end):
                            shared = True
                            break
                        if (v == edge_end and other_tri.vertices[(j + 1) % 3] == edge_start):
                            shared = True
                            break
                
                # If not shared, it's a boundary edge
                if not shared:
                    polygon.append((edge_start, edge_end))
        
        # Remove bad triangles
        for tri in bad_triangles:
            triangles.remove(tri)
        
        # Create new triangles from boundary edges
        for edge_start, edge_end in polygon:
            new_tri = Triangle(edge_start, edge_end, point)
            triangles.append(new_tri)
    
    # Remove triangles that touch the super-triangle vertices
    super_verts = {p1, p2, p3}
    final_triangles = [t for t in triangles 
                      if not any(v in super_verts for v in t.vertices)]
    
    return final_triangles

# --- Visualization ---
def plot_triangulation(triangles, points, title="Delaunay Triangulation"):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot triangles
    for tri in triangles:
        verts = [(v.x, v.y) for v in tri.vertices]
        polygon = Polygon(verts, edgecolor='blue', facecolor='cyan', alpha=0.3, linewidth=1)
        ax.add_patch(polygon)
    
    # Plot Voronoi diagram (dual)
    drawn_edges = set()
    for tri in triangles:
        cc = tri.circumcenter()
        if cc is None:
            continue
        
        # Draw edges to adjacent triangle circumcenters
        for i, v1 in enumerate(tri.vertices):
            v2 = tri.vertices[(i + 1) % 3]
            # Find adjacent triangle
            for other_tri in triangles:
                if other_tri == tri:
                    continue
                # Check if they share edge (v1, v2) or (v2, v1)
                for j, ov1 in enumerate(other_tri.vertices):
                    ov2 = other_tri.vertices[(j + 1) % 3]
                    if (v1 == ov1 and v2 == ov2) or (v1 == ov2 and v2 == ov1):
                        # Found adjacent triangle
                        other_cc = other_tri.circumcenter()
                        if other_cc is not None:
                            edge_key = tuple(sorted([tri.id, other_tri.id]))
                            if edge_key not in drawn_edges:
                                ax.plot([cc.x, other_cc.x], [cc.y, other_cc.y], 'r-', linewidth=1.5)
                                drawn_edges.add(edge_key)
    
    # Plot input points
    px = [p.x for p in points]
    py = [p.y for p in points]
    ax.plot(px, py, 'ko', markersize=6)
    
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlim(min(px)-1, max(px)+1)
    ax.set_ylim(min(py)-1, max(py)+1)
    
    return fig, ax


def plot_voronoi_shaded(triangles, points, title="Voronoi (shaded)"):
    """Shade Voronoi regions by computing each site's cell via half-plane clipping.

    For unweighted Voronoi the power bisector reduces to the perpendicular bisector.
    This function clips a large bounding box by half-planes of the form
    A*x + B*y <= C for each competing site and returns a finite polygon for
    shading. This avoids errors produced by simply connecting circumcenters.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Bounding box (slightly padded)
    all_x = [p.x for p in points]
    all_y = [p.y for p in points]
    xmin, xmax = min(all_x) - 1, max(all_x) + 1
    ymin, ymax = min(all_y) - 1, max(all_y) + 1
    bbox = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

    def clip_polygon_with_halfplane(poly, A, B, C):
        out = []
        if not poly:
            return out
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

    cmap = plt.get_cmap('tab10')
    for i, pi in enumerate(points):
        poly = bbox[:]
        pi_val = pi.x*pi.x + pi.y*pi.y - getattr(pi, 'weight', 0)**2
        for j, pj in enumerate(points):
            if i == j:
                continue
            pj_val = pj.x*pj.x + pj.y*pj.y - getattr(pj, 'weight', 0)**2
            A = 2*(pj.x - pi.x)
            B = 2*(pj.y - pi.y)
            C = pj_val - pi_val
            poly = clip_polygon_with_halfplane(poly, A, B, C)
            if not poly:
                break
        if poly:
            xs, ys = zip(*poly)
            ax.fill(xs, ys, color=cmap(i % 10), alpha=0.25)

    for i, p in enumerate(points):
        ax.plot(p.x, p.y, 'ko')
        ax.text(p.x + 0.08, p.y + 0.08, str(i+1), fontsize=10)

    ax.set_aspect('equal')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    return fig, ax


def generate_random_points(n=8, x_range=(0,10), y_range=(0,10), weight_range=(0.0, 3.0), seed=None):
    """Generate n random Point objects with one-decimal-place coordinates in the
    provided ranges and random weights in weight_range. Returns a list of
    `Point` objects. If `seed` is provided the RNG is seeded for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)
    xs = np.round(np.random.uniform(x_range[0], x_range[1], size=(n,)), 1)
    ys = np.round(np.random.uniform(y_range[0], y_range[1], size=(n,)), 1)
    ws = np.round(np.random.uniform(weight_range[0], weight_range[1], size=(n,)), 2)
    pts = [Point(float(x), float(y), float(w)) for x, y, w in zip(xs, ys, ws)]
    return pts

# --- Main ---
if __name__ == "__main__":
    Point._id_counter = 0
    Triangle._id_counter = 0
    
    # Test data
    point_data = [
        (1, 1), (9, 1), (9, 9), (1, 9),  # Square corners
        (5, 2), (3, 7), (7, 7), (5, 5.1)  # Interior points
    ]
    points = [Point(x, y) for x, y in point_data]
    
    print(f"Computing Delaunay triangulation for {len(points)} points...")
    triangles = bowyer_watson(points)
    
    print(f"Result: {len(triangles)} triangles")
    for tri in triangles:
        verts = [(v.x, v.y) for v in tri.vertices]
        print(f"  {verts}")
    
    # Check which points appear in triangles
    print("\nPoints in triangulation:")
    points_set = set(points)
    for p in points:
        found = False
        for tri in triangles:
            if p in tri.vertices:
                found = True
                break
        status = "✓" if found else "✗"
        print(f"  {status} ({p.x}, {p.y})")
    
    # Show shaded Voronoi (derived from triangle circumcenters)
    fig2, ax2 = plot_voronoi_shaded(triangles, points, "Voronoi Regions (shaded)")
    plt.show()
