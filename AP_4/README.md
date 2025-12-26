# Voronoi and Delaunay for Emergency Facilities

## Problem description
Emergency services (ambulance or fire stations) need to cover a city quickly. A simple way to model coverage is to split the city into service areas where each area is closest to one facility. This helps visualize response regions and the relationships between facilities.

## Mathematical model
- A Voronoi diagram divides the plane into cells. Each cell contains the points that are closest to one facility. This represents the service area of that facility.
- A Delaunay triangulation connects facilities that share a Voronoi boundary. It gives a useful neighbor graph for facility relationships.

## Approach and implementation steps
1. Generate a set of 2D facility locations inside a square city area.
2. Compute the Voronoi diagram using `scipy.spatial.Voronoi`.
3. Compute the Delaunay triangulation using `scipy.spatial.Delaunay`.
4. Optionally generate random population points and assign each point to its nearest facility using a KD-tree.
5. Plot the Voronoi edges, Delaunay edges, facility points, and the assigned population points.

## Experiments
- Number of facilities: 10
- Number of population points: 300
- City bounds: 0 to 100 in both x and y
- Random seeds: 7 for facilities, 21 for population

The plot shows Voronoi edges in orange, Delaunay edges in blue, facility locations as black triangles, and population points colored by their nearest facility.

## Conclusions and limitations
This simple model visualizes service areas and facility neighbors. It does not optimize facility placement and it ignores real city features such as roads, rivers, and traffic. The city is modeled as a flat square region, so results are only illustrative.

## How to run
From the `AP_4` folder:
```bash
python main.py
```
