# osm_batch_router_v2

Rust 批量路由器，为 NC 医疗可达性 pipeline 的 `1_1_build_travel_matrix.py` 与 `0_3_osm_road_quality_audit.py` 提供高性能 OSM routing。

## 正式 router (`src/main.rs`)

输入：

1. 省级 OSM PBF；
2. population grid binary；
3. hospital binary；
4. 输出目录；
5. travel-time cutoff；
6. maximum speed cap；
7. undirected flag；
8. Rayon threads；
9. speed profile list。

Python 会自动调用，不建议手工拼 CLI。

### Snapping

没有 snap-distance gate。

- `motorway`：不进入 edge-snap segment index，只能通过路网 node 接入；
- `motorway_link`：允许 edge snap；
- 其他可驾驶 road segment：允许 edge snap。

对合法 edge snap，router 保存 segment id 与投影 fraction，并按 fraction 把 access cost 分配到路段两端。无需显式复制整张 augmented graph，即可得到与 virtual access point 等价的路径成本；同一 segment 内的 origin/destination 直接计算局部路段成本。

### Routing

- 每个医院一次 bounded Dijkstra，输出 cutoff 内稀疏 hospital-grid travel records；
- 同一 snapped network 另做 multi-source shortest path，输出每个 grid 的 nearest-hospital time；
- snap distance 仅写入 QC，不参与筛除。

## OSM QC helper (`src/bin/road_quality_audit.rs`)

用于独立 `0_3_osm_road_quality_audit.py`。它同时保留：

- raw nearest-node distance；
- raw nearest-edge distance；
- 与正式 router 一致的 legal street access；
- legal access 下的 nearest-hospital routing。

因此 coverage diagnostics 与正式可达性模型可以区分，但 routing 语义保持一致。

## Build

```powershell
cargo build --release --bin osm_batch_router
cargo build --release --bin road_quality_audit
```

Python 入口在发现源码更新时会自动 build。
