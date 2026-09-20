# osm_batch_router_v2

This directory contains the Rust batch router used by the upstream healthcare-accessibility workflow, primarily through `code/1_1_build_travel_matrix.py`.

The public compact reproduction workflow (`python reproduce.py`) does not rebuild national road graphs or travel matrices, so compiling the router is not required for reproducing the released downstream results.

## Router source

The current router is implemented in `src/main.rs` as the `osm_batch_router` binary declared by `Cargo.toml`.

It consumes:

1. an OSM PBF road-network file;
2. a binary population-grid point file;
3. a binary hospital point file;
4. an output directory;
5. a travel-time cutoff;
6. a maximum speed cap;
7. an undirected-routing flag;
8. the Rayon thread count;
9. one or more routing speed profiles;
10. component-rescue settings used by the Python pipeline.

The Python routing stage assembles these arguments automatically; manual invocation is normally unnecessary.

## Snapping semantics

There is no hard snap-distance gate.

- `motorway`: interior edge snapping is disabled; access is through routable network nodes.
- `motorway_link`: edge snapping is allowed.
- other eligible motor-vehicle road segments: edge snapping is allowed.

For a legal edge snap, the router stores the segment ID and projected fraction and distributes access cost to the segment endpoints according to that fraction. This represents a virtual access point without explicitly duplicating the whole augmented graph. Origin/destination pairs snapped to the same segment can use the direct within-segment cost when directionality permits.

Snap distance is retained for quality control but is not itself used as an exclusion threshold.

## Routing outputs

For each configured speed profile, the router:

- performs hospital-centred bounded shortest-path searches and writes sparse hospital-to-grid travel records within the requested cutoff;
- performs a multi-source shortest-path calculation on the same snapped network to obtain nearest-hospital travel time for every reachable grid;
- writes router, road-class, snapping, and connected-component diagnostics used by the upstream Python workflow.

## Build

From this directory:

```bash
cargo build --release
```

The resulting executable is built as `osm_batch_router` (with the platform-appropriate executable suffix). The Python pipeline can also build the binary automatically when required.

## Notes

The router source is provided for transparency and for users who wish to reconstruct the large-scale upstream travel-time workflow from raw OSM and population/hospital inputs. No separate `road_quality_audit` Rust binary is part of the current repository release.
