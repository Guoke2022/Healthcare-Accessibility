# OSM batch router

Rust implementation used by the optional raw reconstruction workflow to compute sparse hospital-to-population-grid travel times on OSM road networks.

The router is called by the Python reconstruction scripts and is not required by the default public `reproduce.py` workflow.

## Inputs

The executable receives a provincial OSM PBF, population-grid binary input, hospital binary input, output directory, travel-time cutoff, maximum speed cap, directionality setting, thread count, and speed-profile configuration.

## Snapping rules

- `motorway`: excluded from edge-segment snapping and accessed through network nodes;
- `motorway_link`: edge snapping is allowed;
- other drivable road segments: edge snapping is allowed.

There is no snap-distance exclusion gate. Snap distance is retained for quality-control reporting.

For valid edge snaps, the router stores the segment identifier and projected fraction and distributes access cost to the segment endpoints. This represents a virtual access point without explicitly duplicating the full road graph.

## Routing outputs

For each hospital, bounded Dijkstra returns sparse travel-time records within the configured cutoff. A separate multi-source shortest-path calculation returns the nearest-hospital travel time for each population grid.

Build the router with:

```bash
cargo build --release --manifest-path osm_batch_router_v2/Cargo.toml
```
