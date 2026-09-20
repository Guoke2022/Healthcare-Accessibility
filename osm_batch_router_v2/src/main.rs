use osmpbfreader::{groups, primitive_block_from_blob, OsmPbfReader};
use rayon::prelude::*;
use rstar::{AABB, PointDistance, RTree, RTreeObject};
use serde::Serialize;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const EARTH_RADIUS_KM: f64 = 6371.0;
const DIST_MULTIPLICATOR: u32 = 262_144;
const INF: u32 = u32::MAX;
const NO_SEGMENT: u32 = u32::MAX;
const SNAP_EDGE: u8 = 0;
const SNAP_NODE: u8 = 1;
const ROUTER_PROTOCOL_VERSION: &str = "component_rescue_v2";

#[derive(Clone, Copy, Debug)]
struct Node { lat: f64, lon: f64 }

#[derive(Clone, Copy, Debug)]
struct RawSegment {
    source: u32,
    target: u32,
    direction: u8, // 0=bidirectional, 1=forward only, 2=reverse only
    legacy_speed: u16,
    new_speed: u16,
    edge_snap_allowed: bool,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
struct ComponentStat {
    road_length_km: f64,
    segment_count: u64,
    node_count: u64,
    raw_grid_count: u64,
    raw_hospital_count: u64,
}

#[derive(Clone, Copy, Debug)]
struct AdjEdge { to: u32, w_legacy: u32, w_new: u32 }

#[derive(Clone, Copy, Debug)]
struct GridPoint { grid_id: i64, lon: f64, lat: f64 }
#[derive(Clone, Copy, Debug)]
struct HospitalPoint { hospital_row: i32, lon: f64, lat: f64 }

#[derive(Clone, Copy, Debug)]
struct SnapAccess {
    kind: u8,
    node_id: u32,
    segment_id: u32,
    fraction: f32,
    snap_km: f32,
    component_id: u32,
}

#[derive(Clone, Copy, Debug)]
struct GridSnap { grid_id: i64, access: SnapAccess }
#[derive(Clone, Copy, Debug)]
struct HospitalSnap { hospital_row: i32, access: SnapAccess }

#[derive(Clone, Copy)]
struct SpatialNode { xyz: [f64; 3], id: u32 }
impl RTreeObject for SpatialNode {
    type Envelope = AABB<[f64; 3]>;
    fn envelope(&self) -> Self::Envelope { AABB::from_point(self.xyz) }
}
impl PointDistance for SpatialNode {
    fn distance_2(&self, p: &[f64; 3]) -> f64 {
        let dx=self.xyz[0]-p[0]; let dy=self.xyz[1]-p[1]; let dz=self.xyz[2]-p[2];
        dx*dx+dy*dy+dz*dz
    }
}

#[derive(Clone, Copy)]
struct SpatialSegment { a: [f64; 3], b: [f64; 3], id: u32 }
impl RTreeObject for SpatialSegment {
    type Envelope = AABB<[f64; 3]>;
    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners(
            [self.a[0].min(self.b[0]), self.a[1].min(self.b[1]), self.a[2].min(self.b[2])],
            [self.a[0].max(self.b[0]), self.a[1].max(self.b[1]), self.a[2].max(self.b[2])],
        )
    }
}
impl PointDistance for SpatialSegment {
    fn distance_2(&self, p: &[f64; 3]) -> f64 {
        let (_, d2) = project_fraction_3d(*p, self.a, self.b);
        d2
    }
}

#[derive(Default, Serialize, Clone)]
struct RoadClassStat {
    way_count: u64,
    segment_count: u64,
    included_way_count: u64,
    included_segment_count: u64,
    explicit_maxspeed_way_count: u64,
    fallback_speed_way_count: u64,
    legacy_default_speed_kmh: u32,
    chn_osm_default_speed_kmh: u32,
}

#[derive(Serialize)]
struct ProfileSummary {
    profile: String,
    output_file: String,
    nearest_output_file: String,
    route_searches: u64,
    skipped_hospitals_no_target_component: u64,
    travel_records: u64,
    nearest_reachable_grids: u64,
    routing_seconds: f64,
    nearest_seconds: f64,
}

#[derive(Serialize)]
struct RouterSummary {
    pbf: String,
    grid_count: usize,
    hospital_count: usize,
    node_count: usize,
    raw_segment_count: usize,
    directed_edge_count: usize,
    undirected: bool,
    cutoff_min: f64,
    max_speed_kmh: u32,
    snap_gate_applied: bool,
    snap_semantics: String,
    motorway_edge_snap_allowed: bool,
    motorway_link_edge_snap_allowed: bool,
    component_rescue_enabled: bool,
    tiny_component_max_road_km: f64,
    component_rescue_min_size_ratio: f64,
    component_rescue_min_grid_count: u64,
    component_rescue_min_grid_ratio: f64,
    component_rescue_max_extra_km: f64,
    component_count: usize,
    tiny_component_count: usize,
    sparse_grid_component_count: usize,
    suspicious_component_count: usize,
    grid_component_rescue_count: usize,
    hospital_component_rescue_count: usize,
    grid_edge_snap_count: usize,
    grid_node_snap_count: usize,
    hospital_edge_snap_count: usize,
    hospital_node_snap_count: usize,
    parse_seconds: f64,
    graph_seconds: f64,
    component_seconds: f64,
    snap_seconds: f64,
    raw_max_speed_kmh_seen: u32,
    speed_capped_legacy_way_count: u64,
    speed_capped_chn_osm_default_way_count: u64,
    profiles: Vec<ProfileSummary>,
}

fn usage() -> ! {
    eprintln!("Usage:\nosm_batch_router <pbf> <grids.bin> <hospitals.bin> <output_dir> <cutoff_min> <max_speed_kmh> <undirected:true|false> <threads> <profiles> <component_rescue:true|false> <tiny_component_max_road_km> <rescue_min_size_ratio> <rescue_min_grid_count> <rescue_min_grid_ratio> <rescue_max_extra_km>");
    std::process::exit(2);
}

fn lonlat_to_xyz(lon:f64, lat:f64)->[f64;3]{
    let lo=lon.to_radians(); let la=lat.to_radians(); let c=la.cos();
    [c*lo.cos(), c*lo.sin(), la.sin()]
}
fn chord_to_km(chord:f64)->f64 { 2.0*(chord/2.0).clamp(0.0,1.0).asin()*EARTH_RADIUS_KM }
fn haversine_km(a:Node,b:Node)->f64{
    let dlat=(b.lat-a.lat).to_radians(); let dlon=(b.lon-a.lon).to_radians();
    let lat1=a.lat.to_radians(); let lat2=b.lat.to_radians();
    let x=(dlat/2.0).sin().powi(2)+(dlon/2.0).sin().powi(2)*lat1.cos()*lat2.cos();
    2.0*x.sqrt().atan2((1.0-x).sqrt())*EARTH_RADIUS_KM
}
fn project_fraction_3d(p:[f64;3],a:[f64;3],b:[f64;3])->(f64,f64){
    let ab=[b[0]-a[0],b[1]-a[1],b[2]-a[2]];
    let ap=[p[0]-a[0],p[1]-a[1],p[2]-a[2]];
    let den=ab[0]*ab[0]+ab[1]*ab[1]+ab[2]*ab[2];
    let t=if den<=1e-18 {0.0} else {((ap[0]*ab[0]+ap[1]*ab[1]+ap[2]*ab[2])/den).clamp(0.0,1.0)};
    let q=[a[0]+t*ab[0],a[1]+t*ab[1],a[2]+t*ab[2]];
    let dx=p[0]-q[0]; let dy=p[1]-q[1]; let dz=p[2]-q[2];
    (t,dx*dx+dy*dy+dz*dz)
}

fn old_resolve_max_speed(s:&str)->Option<u32>{match s{
    "DE:motorway"=>Some(120),"DE:rural"|"AT:rural"=>Some(100),"DE:urban"|"AT:urban"|"CZ:urban"=>Some(50),
    "maxspeed=50"=>Some(50),"50;"|"50b"=>Some(50),"DE:living_street"=>Some(30),"30 kph"=>Some(30),
    "zone:maxspeed=de:30"|"DE:zone:30"|"DE:zone30"=>Some(30),"30 mph"=>Some(48),"20:forward"=>Some(20),
    "10 mph"=>Some(16),"5 mph"=>Some(8),"DE:walk"|"walk"|"Schrittgeschwindigkeit"=>Some(7),_=>None}}
fn legacy_default_speed(h:&str)->u32{match h{
    "motorway"=>120,"motorway_link"=>60,"trunk"=>100,"trunk_link"=>50,"primary"=>60,"primary_link"=>50,
    "secondary"|"secondary_link"=>50,"tertiary"|"tertiary_link"=>50,"unclassified"=>40,"residential"=>30,
    "track"|"service"=>10,"living_street"=>7,"path"|"walk"|"pedestrian"|"footway"=>4,_=>50}}
fn chn_osm_default_speed(h:&str)->u32{match h{
    "motorway"=>120,"motorway_link"=>50,"trunk"=>80,"trunk_link"=>40,"primary"=>60,"primary_link"=>50,
    "secondary"=>50,"secondary_link"=>40,"tertiary"|"tertiary_link"=>40,"unclassified"=>40,"road"=>30,
    "residential"|"service"|"living_street"=>20,"track"=>10,_=>40}}
fn car_allowed(h:&str)->bool{matches!(h,"motorway"|"motorway_link"|"trunk"|"trunk_link"|"primary"|"primary_link"|"secondary"|"secondary_link"|"tertiary"|"tertiary_link"|"unclassified"|"residential"|"living_street"|"service"|"road")}
fn is_explicit_no(s:&str)->bool{matches!(s.trim().to_ascii_lowercase().as_str(),"no"|"private"|"use_sidepath")}
fn parse_oneway(oneway:&str,junction:&str)->u8{match oneway.trim().to_ascii_lowercase().as_str(){"yes"|"1"|"true"=>1,"-1"|"reverse"=>2,"no"|"0"|"false"=>0,_ if junction.trim().eq_ignore_ascii_case("roundabout")=>1,_=>0}}
fn parse_speed_kmh(s:&str)->Option<u32>{
    let v=s.trim(); if v.is_empty(){return None;} if let Ok(x)=v.parse::<f64>(){return (x>0.0).then_some(x.round() as u32)}
    let l=v.to_ascii_lowercase(); for suf in [" km/h"," kmh"," kph"]{if let Some(x)=l.strip_suffix(suf){if let Ok(y)=x.trim().parse::<f64>(){return (y>0.0).then_some(y.round() as u32)}}}
    if let Some(x)=l.strip_suffix(" mph"){if let Ok(y)=x.trim().parse::<f64>(){return (y>0.0).then_some((y*1.609_344).round() as u32)}} old_resolve_max_speed(v)
}
fn parse_speed_pair(maxspeed:&str,h:&str,cap:u32)->(u16,u16,bool,bool,u32){
    let e=parse_speed_kmh(maxspeed); let a=e.unwrap_or_else(||legacy_default_speed(h)).max(1); let b=e.unwrap_or_else(||chn_osm_default_speed(h)).max(1);
    (a.min(cap) as u16,b.min(cap) as u16,a>cap,b>cap,a.max(b))
}
fn get_or_insert_node(map:&mut HashMap<i64,u32>,osm_id:i64)->u32{if let Some(&x)=map.get(&osm_id){x}else{let x=map.len() as u32;map.insert(osm_id,x);x}}

fn parse_pbf(pbf_path:&Path,max_speed_kmh:u32)->Result<(Vec<Node>,Vec<RawSegment>,BTreeMap<String,RoadClassStat>,u32,u64,u64),Box<dyn std::error::Error>>{
    let mut node_map=HashMap::new(); let mut segments=Vec::new(); let mut stats:BTreeMap<String,RoadClassStat>=BTreeMap::new();
    let mut raw_max=0; let mut cap_l=0; let mut cap_n=0; let file=File::open(pbf_path)?; let mut pbf=OsmPbfReader::new(file);
    for blob in pbf.blobs(){let block=primitive_block_from_blob(&blob?)?; for group in block.get_primitivegroup(){for way in groups::ways(group,&block){
        if !way.tags.contains_key("highway")||way.nodes.len()<2{continue;} let highway=way.tags.get("highway").unwrap().trim().to_string(); let nseg=(way.nodes.len()-1) as u64;
        let access=["motorcar","motor_vehicle","vehicle","access"].iter().find_map(|k|way.tags.get(*k)); let denied=access.map(|v|is_explicit_no(v)).unwrap_or(false); let allowed=car_allowed(&highway)&&!denied;
        let maxspeed=way.tags.get("maxspeed").map(|s|s.trim()).unwrap_or(""); let explicit=parse_speed_kmh(maxspeed).is_some(); let (ls,ns,cl,cn,rm)=parse_speed_pair(maxspeed,&highway,max_speed_kmh);
        raw_max=raw_max.max(rm); if cl{cap_l+=1}; if cn{cap_n+=1}; let e=stats.entry(highway.clone()).or_default(); e.way_count+=1;e.segment_count+=nseg;e.legacy_default_speed_kmh=legacy_default_speed(&highway);e.chn_osm_default_speed_kmh=chn_osm_default_speed(&highway);if explicit{e.explicit_maxspeed_way_count+=1}else{e.fallback_speed_way_count+=1}; if !allowed{continue;} e.included_way_count+=1;e.included_segment_count+=nseg;
        let direction=parse_oneway(way.tags.get("oneway").map(|s|s.trim()).unwrap_or(""),way.tags.get("junction").map(|s|s.trim()).unwrap_or("")); let mut prev=get_or_insert_node(&mut node_map,way.nodes[0].0);
        for n in way.nodes.iter().skip(1){let cur=get_or_insert_node(&mut node_map,n.0);segments.push(RawSegment{source:prev,target:cur,direction,legacy_speed:ls,new_speed:ns,edge_snap_allowed:highway!="motorway"});prev=cur;}
    }}}
    pbf.rewind()?; let mut nodes=vec![Node{lat:f64::NAN,lon:f64::NAN};node_map.len()];
    for blob in pbf.blobs(){let block=primitive_block_from_blob(&blob?)?;for group in block.get_primitivegroup(){for n in groups::dense_nodes(group,&block){if let Some(&idx)=node_map.get(&n.id.0){nodes[idx as usize]=Node{lat:n.decimicro_lat as f64/1e7,lon:n.decimicro_lon as f64/1e7};}}}}
    let missing=nodes.iter().filter(|n|!n.lat.is_finite()||!n.lon.is_finite()).count(); if missing>0{return Err(format!("{} road nodes missing coordinates",missing).into())}
    Ok((nodes,segments,stats,raw_max,cap_l,cap_n))
}

fn edge_weight(distance_km:f64,speed:u16)->u32{((distance_km*DIST_MULTIPLICATOR as f64) as u32)/(speed.max(1) as u32)}
fn frac_weight(w:u32,f:f64)->u32{((w as f64*f.clamp(0.0,1.0)).round() as u64).min(u32::MAX as u64) as u32}
fn segment_weights(seg:&RawSegment,nodes:&[Node])->(u32,u32){let d=haversine_km(nodes[seg.source as usize],nodes[seg.target as usize]);(edge_weight(d,seg.legacy_speed),edge_weight(d,seg.new_speed))}

fn build_graph(nodes:&[Node],segments:&[RawSegment],undirected:bool)->(Vec<usize>,Vec<AdjEdge>,usize){
    let n=nodes.len(); let mut degree=vec![0usize;n];let mut edge_count=0; for s in segments{if undirected{degree[s.source as usize]+=1;degree[s.target as usize]+=1;edge_count+=2}else{match s.direction{1=>{degree[s.source as usize]+=1;edge_count+=1},2=>{degree[s.target as usize]+=1;edge_count+=1},_=>{degree[s.source as usize]+=1;degree[s.target as usize]+=1;edge_count+=2}}}}
    let mut offsets=vec![0usize;n+1];for i in 0..n{offsets[i+1]=offsets[i]+degree[i];} let mut cursor=offsets[..n].to_vec();let mut edges=vec![AdjEdge{to:0,w_legacy:0,w_new:0};edge_count];
    let mut put=|from:u32,to:u32,wl:u32,wn:u32|{let idx=cursor[from as usize];edges[idx]=AdjEdge{to,w_legacy:wl,w_new:wn};cursor[from as usize]+=1;};
    for s in segments{let (wl,wn)=segment_weights(s,nodes);if undirected{put(s.source,s.target,wl,wn);put(s.target,s.source,wl,wn)}else{match s.direction{1=>put(s.source,s.target,wl,wn),2=>put(s.target,s.source,wl,wn),_=>{put(s.source,s.target,wl,wn);put(s.target,s.source,wl,wn)}}}}
    (offsets,edges,edge_count)
}
fn build_reverse_graph(node_count:usize,offsets:&[usize],edges:&[AdjEdge])->(Vec<usize>,Vec<AdjEdge>){
    let mut deg=vec![0usize;node_count];for from in 0..node_count{for e in &edges[offsets[from]..offsets[from+1]]{deg[e.to as usize]+=1;}}let mut ro=vec![0usize;node_count+1];for i in 0..node_count{ro[i+1]=ro[i]+deg[i];}let mut cur=ro[..node_count].to_vec();let mut re=vec![AdjEdge{to:0,w_legacy:0,w_new:0};edges.len()];for from in 0..node_count{for &e in &edges[offsets[from]..offsets[from+1]]{let idx=cur[e.to as usize];re[idx]=AdjEdge{to:from as u32,w_legacy:e.w_legacy,w_new:e.w_new};cur[e.to as usize]+=1;}}(ro,re)
}

struct UnionFind{parent:Vec<u32>,rank:Vec<u8>}
impl UnionFind{fn new(n:usize)->Self{Self{parent:(0..n as u32).collect(),rank:vec![0;n]}}fn find(&mut self,x:u32)->u32{let p=self.parent[x as usize];if p!=x{let r=self.find(p);self.parent[x as usize]=r;}self.parent[x as usize]}fn union(&mut self,a:u32,b:u32){let mut ra=self.find(a);let mut rb=self.find(b);if ra==rb{return;}let ka=self.rank[ra as usize];let kb=self.rank[rb as usize];if ka<kb{std::mem::swap(&mut ra,&mut rb)}self.parent[rb as usize]=ra;if ka==kb{self.rank[ra as usize]+=1}}}
fn compute_components(n:usize,segments:&[RawSegment])->Vec<u32>{let mut uf=UnionFind::new(n);for s in segments{uf.union(s.source,s.target)}let mut map=HashMap::new();let mut next=0;let mut out=vec![0;n];for i in 0..n as u32{let r=uf.find(i);let c=*map.entry(r).or_insert_with(||{let x=next;next+=1;x});out[i as usize]=c;}out}

fn compute_component_stats(nodes:&[Node],segments:&[RawSegment],components:&[u32])->Vec<ComponentStat>{
    let n_comp=components.iter().copied().max().map(|x|x as usize+1).unwrap_or(0);
    let mut stats=vec![ComponentStat::default();n_comp];
    for &c in components{stats[c as usize].node_count+=1;}
    for s in segments{
        let c=components[s.source as usize] as usize;
        debug_assert_eq!(components[s.source as usize],components[s.target as usize]);
        stats[c].segment_count+=1;
        stats[c].road_length_km+=haversine_km(nodes[s.source as usize],nodes[s.target as usize]);
    }
    stats
}

fn write_component_stats(path:&Path,stats:&[ComponentStat],tiny_max_road_km:f64,min_grid_count:u64)->std::io::Result<()>{
    let mut w=BufWriter::new(File::create(path)?);
    writeln!(w,"component_id,road_length_km,segment_count,node_count,raw_grid_count,raw_hospital_count,is_tiny,is_sparse_grid,is_suspicious")?;
    for (cid,s) in stats.iter().enumerate(){
        let tiny=s.road_length_km<tiny_max_road_km;
        let sparse=s.raw_grid_count<min_grid_count;
        writeln!(w,"{},{:.6},{},{},{},{},{},{},{}",cid,s.road_length_km,s.segment_count,s.node_count,s.raw_grid_count,s.raw_hospital_count,tiny,sparse,tiny||sparse)?;
    }
    w.flush()
}

fn read_grid_points(path:&Path)->Result<Vec<GridPoint>,Box<dyn std::error::Error>>{let mut b=Vec::new();File::open(path)?.read_to_end(&mut b)?;if b.len()%24!=0{return Err("invalid grid binary size".into())}Ok(b.chunks_exact(24).map(|c|GridPoint{grid_id:i64::from_le_bytes(c[0..8].try_into().unwrap()),lon:f64::from_le_bytes(c[8..16].try_into().unwrap()),lat:f64::from_le_bytes(c[16..24].try_into().unwrap())}).collect())}
fn read_hospital_points(path:&Path)->Result<Vec<HospitalPoint>,Box<dyn std::error::Error>>{let mut b=Vec::new();File::open(path)?.read_to_end(&mut b)?;if b.len()%20!=0{return Err("invalid hospital binary size".into())}Ok(b.chunks_exact(20).map(|c|HospitalPoint{hospital_row:i32::from_le_bytes(c[0..4].try_into().unwrap()),lon:f64::from_le_bytes(c[4..12].try_into().unwrap()),lat:f64::from_le_bytes(c[12..20].try_into().unwrap())}).collect())}

fn snap_one_raw(lon:f64,lat:f64,segments:&[RawSegment],components:&[u32],node_tree:&RTree<SpatialNode>,seg_tree:&RTree<SpatialSegment>)->SnapAccess{
    let p=lonlat_to_xyz(lon,lat);let nn=node_tree.nearest_neighbor(&p).expect("empty road node tree");let node_d2=nn.distance_2(&p);
    if let Some(ss)=seg_tree.nearest_neighbor(&p){let (t,seg_d2)=project_fraction_3d(p,ss.a,ss.b);if seg_d2<=node_d2{let s=segments[ss.id as usize];return SnapAccess{kind:SNAP_EDGE,node_id:s.source,segment_id:ss.id,fraction:t as f32,snap_km:chord_to_km(seg_d2.sqrt()) as f32,component_id:components[s.source as usize]};}}
    SnapAccess{kind:SNAP_NODE,node_id:nn.id,segment_id:NO_SEGMENT,fraction:0.0,snap_km:chord_to_km(node_d2.sqrt()) as f32,component_id:components[nn.id as usize]}
}

fn raw_component_suspicious(raw_component:u32,component_stats:&[ComponentStat],tiny_max_road_km:f64,min_grid_count:u64)->bool{
    let s=&component_stats[raw_component as usize];
    s.road_length_km<tiny_max_road_km || s.raw_grid_count<min_grid_count
}

fn rescue_component_eligible(raw_component:u32,alt_component:u32,component_stats:&[ComponentStat],tiny_max_road_km:f64,min_size_ratio:f64,min_grid_count:u64,min_grid_ratio:f64)->bool{
    if alt_component==raw_component{return false;}
    let raw=&component_stats[raw_component as usize];let alt=&component_stats[alt_component as usize];
    if alt.road_length_km<tiny_max_road_km || alt.raw_grid_count<min_grid_count{return false;}
    let road_ok=alt.road_length_km>=raw.road_length_km.max(1e-6)*min_size_ratio;
    let grid_need=((raw.raw_grid_count as f64)*min_grid_ratio).ceil() as u64;
    let grid_ok=alt.raw_grid_count>=grid_need.max(min_grid_count);
    road_ok || grid_ok
}

fn nearest_rescue_candidate(lon:f64,lat:f64,raw:SnapAccess,segments:&[RawSegment],components:&[u32],component_stats:&[ComponentStat],node_tree:&RTree<SpatialNode>,seg_tree:&RTree<SpatialSegment>,tiny_max_road_km:f64,min_size_ratio:f64,min_grid_count:u64,min_grid_ratio:f64,max_extra_km:f64)->Option<SnapAccess>{
    let p=lonlat_to_xyz(lon,lat);let max_snap_km=raw.snap_km as f64+max_extra_km;let mut best:Option<(f64,SnapAccess)>=None;
    for nn in node_tree.nearest_neighbor_iter(&p){
        let d2=nn.distance_2(&p);let dkm=chord_to_km(d2.sqrt());if dkm>max_snap_km{break;}
        let c=components[nn.id as usize];if !rescue_component_eligible(raw.component_id,c,component_stats,tiny_max_road_km,min_size_ratio,min_grid_count,min_grid_ratio){continue;}
        best=Some((d2,SnapAccess{kind:SNAP_NODE,node_id:nn.id,segment_id:NO_SEGMENT,fraction:0.0,snap_km:dkm as f32,component_id:c}));break;
    }
    for ss in seg_tree.nearest_neighbor_iter(&p){
        let (t,d2)=project_fraction_3d(p,ss.a,ss.b);let dkm=chord_to_km(d2.sqrt());if dkm>max_snap_km{break;}
        let seg=segments[ss.id as usize];let c=components[seg.source as usize];if !rescue_component_eligible(raw.component_id,c,component_stats,tiny_max_road_km,min_size_ratio,min_grid_count,min_grid_ratio){continue;}
        if best.map(|x|d2<=x.0).unwrap_or(true){best=Some((d2,SnapAccess{kind:SNAP_EDGE,node_id:seg.source,segment_id:ss.id,fraction:t as f32,snap_km:dkm as f32,component_id:c}));}
        break;
    }
    best.map(|x|x.1)
}

fn rescue_from_raw(lon:f64,lat:f64,raw:SnapAccess,segments:&[RawSegment],components:&[u32],component_stats:&[ComponentStat],node_tree:&RTree<SpatialNode>,seg_tree:&RTree<SpatialSegment>,rescue_enabled:bool,tiny_max_road_km:f64,min_size_ratio:f64,min_grid_count:u64,min_grid_ratio:f64,max_extra_km:f64)->(SnapAccess,bool){
    if !rescue_enabled || !raw_component_suspicious(raw.component_id,component_stats,tiny_max_road_km,min_grid_count){return (raw,false);}
    if let Some(alt)=nearest_rescue_candidate(lon,lat,raw,segments,components,component_stats,node_tree,seg_tree,tiny_max_road_km,min_size_ratio,min_grid_count,min_grid_ratio,max_extra_km){
        if alt.snap_km as f64<=raw.snap_km as f64+max_extra_km{return (alt,true);}
    }
    (raw,false)
}

fn snap_points(nodes:&[Node],segments:&[RawSegment],components:&[u32],component_stats:&mut [ComponentStat],grids:&[GridPoint],hospitals:&[HospitalPoint],rescue_enabled:bool,tiny_max_road_km:f64,min_size_ratio:f64,min_grid_count:u64,min_grid_ratio:f64,max_extra_km:f64)->(Vec<GridSnap>,Vec<HospitalSnap>,usize,usize){
    let node_tree=RTree::bulk_load(nodes.par_iter().enumerate().map(|(id,n)|SpatialNode{xyz:lonlat_to_xyz(n.lon,n.lat),id:id as u32}).collect());
    let seg_tree=RTree::bulk_load(segments.par_iter().enumerate().filter(|(_,s)|s.edge_snap_allowed).map(|(id,s)|SpatialSegment{a:lonlat_to_xyz(nodes[s.source as usize].lon,nodes[s.source as usize].lat),b:lonlat_to_xyz(nodes[s.target as usize].lon,nodes[s.target as usize].lat),id:id as u32}).collect());

    // Two-pass design: first snap every populated grid/hospital using pure nearest legal street access.
    // These raw assignments measure how much populated-grid support each component actually has,
    // without using beds, travel-time results, R, or any accessibility output.
    let raw_gs:Vec<GridSnap>=grids.par_iter().map(|p|GridSnap{grid_id:p.grid_id,access:snap_one_raw(p.lon,p.lat,segments,components,&node_tree,&seg_tree)}).collect();
    let raw_hs:Vec<HospitalSnap>=hospitals.par_iter().map(|p|HospitalSnap{hospital_row:p.hospital_row,access:snap_one_raw(p.lon,p.lat,segments,components,&node_tree,&seg_tree)}).collect();
    for s in component_stats.iter_mut(){s.raw_grid_count=0;s.raw_hospital_count=0;}
    for g in &raw_gs{component_stats[g.access.component_id as usize].raw_grid_count+=1;}
    for h in &raw_hs{component_stats[h.access.component_id as usize].raw_hospital_count+=1;}

    // Second pass: only suspicious components (short road length OR very few populated grids)
    // are eligible for rescue, and the destination must itself have adequate populated-grid support.
    let grid_rescued=AtomicU64::new(0);let hospital_rescued=AtomicU64::new(0);
    let gs:Vec<GridSnap>=grids.par_iter().zip(raw_gs.par_iter()).map(|(p,r)|{let (access,rescued)=rescue_from_raw(p.lon,p.lat,r.access,segments,components,component_stats,&node_tree,&seg_tree,rescue_enabled,tiny_max_road_km,min_size_ratio,min_grid_count,min_grid_ratio,max_extra_km);if rescued{grid_rescued.fetch_add(1,Ordering::Relaxed);}GridSnap{grid_id:p.grid_id,access}}).collect();
    let hs:Vec<HospitalSnap>=hospitals.par_iter().zip(raw_hs.par_iter()).map(|(p,r)|{let (access,rescued)=rescue_from_raw(p.lon,p.lat,r.access,segments,components,component_stats,&node_tree,&seg_tree,rescue_enabled,tiny_max_road_km,min_size_ratio,min_grid_count,min_grid_ratio,max_extra_km);if rescued{hospital_rescued.fetch_add(1,Ordering::Relaxed);}HospitalSnap{hospital_row:p.hospital_row,access}}).collect();
    (gs,hs,grid_rescued.load(Ordering::Relaxed) as usize,hospital_rescued.load(Ordering::Relaxed) as usize)
}
fn write_grid_snap(path:&Path,rows:&[GridSnap])->std::io::Result<()>{let mut w=BufWriter::new(File::create(path)?);for r in rows{w.write_all(&r.grid_id.to_le_bytes())?;w.write_all(&[r.access.kind])?;w.write_all(&r.access.node_id.to_le_bytes())?;w.write_all(&r.access.segment_id.to_le_bytes())?;w.write_all(&r.access.fraction.to_le_bytes())?;w.write_all(&r.access.snap_km.to_le_bytes())?;w.write_all(&r.access.component_id.to_le_bytes())?;}Ok(())}
fn write_hospital_snap(path:&Path,rows:&[HospitalSnap])->std::io::Result<()>{let mut w=BufWriter::new(File::create(path)?);for r in rows{w.write_all(&r.hospital_row.to_le_bytes())?;w.write_all(&[r.access.kind])?;w.write_all(&r.access.node_id.to_le_bytes())?;w.write_all(&r.access.segment_id.to_le_bytes())?;w.write_all(&r.access.fraction.to_le_bytes())?;w.write_all(&r.access.snap_km.to_le_bytes())?;w.write_all(&r.access.component_id.to_le_bytes())?;}Ok(())}

#[derive(Clone,Copy)] enum Profile{Legacy,ChnOsmDefault}
impl Profile{fn name(&self)->&'static str{match self{Profile::Legacy=>"legacy",Profile::ChnOsmDefault=>"chn_osm_default"}}fn weight(&self,e:AdjEdge)->u32{match self{Profile::Legacy=>e.w_legacy,Profile::ChnOsmDefault=>e.w_new}}fn segment_weight(&self,s:&RawSegment,nodes:&[Node])->u32{let (a,b)=segment_weights(s,nodes);match self{Profile::Legacy=>a,Profile::ChnOsmDefault=>b}}}
fn parse_profiles(s:&str)->Vec<Profile>{let mut out=Vec::new();for x in s.split(','){match x.trim(){"legacy"=>out.push(Profile::Legacy),"chn_osm_default"=>out.push(Profile::ChnOsmDefault),""=>{},_=>usage()}}if out.is_empty(){usage()}out}

#[derive(Clone,Copy)]struct TargetEntry{grid_idx:usize,connector:u32}
fn target_connectors(access:SnapAccess,seg:&RawSegment,w:u32,undirected:bool)->Vec<(u32,u32)>{
    if access.kind==SNAP_NODE{return vec![(access.node_id,0)]}let t=access.fraction as f64;let a=frac_weight(w,t);let b=frac_weight(w,1.0-t);if undirected{return vec![(seg.source,a),(seg.target,b)]}match seg.direction{1=>vec![(seg.source,a)],2=>vec![(seg.target,b)],_=>vec![(seg.source,a),(seg.target,b)]}
}
fn origin_connectors(access:SnapAccess,seg:&RawSegment,w:u32,undirected:bool)->Vec<(u32,u32)>{
    if access.kind==SNAP_NODE{return vec![(access.node_id,0)]}let t=access.fraction as f64;let a=frac_weight(w,t);let b=frac_weight(w,1.0-t);if undirected{return vec![(seg.source,a),(seg.target,b)]}match seg.direction{1=>vec![(seg.target,b)],2=>vec![(seg.source,a)],_=>vec![(seg.source,a),(seg.target,b)]}
}
fn direct_same_segment_cost(a:f32,b:f32,seg:&RawSegment,w:u32,undirected:bool)->Option<u32>{let da=a as f64;let db=b as f64;if undirected{return Some(frac_weight(w,(da-db).abs()))}match seg.direction{1 if db>=da=>Some(frac_weight(w,db-da)),2 if db<=da=>Some(frac_weight(w,da-db)),0=>Some(frac_weight(w,(da-db).abs())),_=>None}}

fn build_target_index(profile:Profile,nodes:&[Node],segments:&[RawSegment],grid_snaps:&[GridSnap],node_count:usize,undirected:bool)->(Vec<usize>,Vec<TargetEntry>,Vec<usize>,Vec<usize>,HashSet<u32>){
    let mut node_pairs:Vec<(u32,TargetEntry)>=Vec::new();let mut seg_pairs:Vec<(u32,usize)>=Vec::new();let mut comps=HashSet::new();
    for (idx,g) in grid_snaps.iter().enumerate(){comps.insert(g.access.component_id);if g.access.kind==SNAP_NODE{node_pairs.push((g.access.node_id,TargetEntry{grid_idx:idx,connector:0}))}else{let s=&segments[g.access.segment_id as usize];let w=profile.segment_weight(s,nodes);for (n,c) in target_connectors(g.access,s,w,undirected){node_pairs.push((n,TargetEntry{grid_idx:idx,connector:c}))}seg_pairs.push((g.access.segment_id,idx));}}
    node_pairs.sort_unstable_by_key(|x|x.0);let mut counts=vec![0usize;node_count];for (n,_) in &node_pairs{counts[*n as usize]+=1}let mut offsets=vec![0usize;node_count+1];for i in 0..node_count{offsets[i+1]=offsets[i]+counts[i]}let entries=node_pairs.into_iter().map(|x|x.1).collect();
    seg_pairs.sort_unstable_by_key(|x|x.0);let mut sc=vec![0usize;segments.len()];for (s,_) in &seg_pairs{sc[*s as usize]+=1}let mut so=vec![0usize;segments.len()+1];for i in 0..segments.len(){so[i+1]=so[i]+sc[i]}let si=seg_pairs.into_iter().map(|x|x.1).collect();(offsets,entries,so,si,comps)
}

struct Workspace{dist:Vec<u32>,stamp:Vec<u32>,cur:u32,heap:BinaryHeap<(Reverse<u32>,u32)>}
impl Workspace{fn new(n:usize)->Self{Self{dist:vec![0;n],stamp:vec![0;n],cur:0,heap:BinaryHeap::new()}}fn begin(&mut self){self.cur=self.cur.wrapping_add(1);if self.cur==0{self.stamp.fill(0);self.cur=1}self.heap.clear()}fn get(&self,n:u32)->u32{if self.stamp[n as usize]==self.cur{self.dist[n as usize]}else{INF}}fn set_min(&mut self,n:u32,v:u32){if v<self.get(n){self.stamp[n as usize]=self.cur;self.dist[n as usize]=v;self.heap.push((Reverse(v),n));}}}

fn route_one(ws:&mut Workspace,h:&HospitalSnap,profile:Profile,cutoff:u32,nodes:&[Node],segments:&[RawSegment],offsets:&[usize],edges:&[AdjEdge],target_offsets:&[usize],targets:&[TargetEntry],seg_offsets:&[usize],seg_grid_indices:&[usize],grid_snaps:&[GridSnap],grids:&[GridPoint],undirected:bool)->Vec<(i32,i64,f32)>{
    ws.begin();let mut best:HashMap<usize,u32>=HashMap::new();
    if h.access.kind==SNAP_NODE{ws.set_min(h.access.node_id,0)}else{let s=&segments[h.access.segment_id as usize];let w=profile.segment_weight(s,nodes);for (n,c) in origin_connectors(h.access,s,w,undirected){if c<=cutoff{ws.set_min(n,c)}}let a=seg_offsets[h.access.segment_id as usize];let b=seg_offsets[h.access.segment_id as usize+1];for &gi in &seg_grid_indices[a..b]{if let Some(c)=direct_same_segment_cost(h.access.fraction,grid_snaps[gi].access.fraction,s,w,undirected){if c<=cutoff{best.entry(gi).and_modify(|x|*x=(*x).min(c)).or_insert(c);}}}}
    while let Some((Reverse(cost),node))=ws.heap.pop(){if cost!=ws.get(node){continue}if cost>cutoff{break}for te in &targets[target_offsets[node as usize]..target_offsets[node as usize+1]]{let c=cost.saturating_add(te.connector);if c<=cutoff{best.entry(te.grid_idx).and_modify(|x|*x=(*x).min(c)).or_insert(c);}}for &e in &edges[offsets[node as usize]..offsets[node as usize+1]]{let c=cost.saturating_add(profile.weight(e));if c<=cutoff{ws.set_min(e.to,c)}}}
    let mut rows=Vec::with_capacity(best.len());for (gi,c) in best{rows.push((h.hospital_row,grids[gi].grid_id,(c as f64/DIST_MULTIPLICATOR as f64*60.0) as f32));}rows
}
fn write_travel_records(writer:&Mutex<BufWriter<File>>,rows:&[(i32,i64,f32)])->std::io::Result<()>{let mut w=writer.lock().unwrap();for (h,g,t) in rows{w.write_all(&h.to_le_bytes())?;w.write_all(&g.to_le_bytes())?;w.write_all(&t.to_le_bytes())?;}Ok(())}

fn build_hospitals_by_segment(hs:&[HospitalSnap])->HashMap<u32,Vec<f32>>{let mut m:HashMap<u32,Vec<f32>>=HashMap::new();for h in hs{if h.access.kind==SNAP_EDGE{m.entry(h.access.segment_id).or_default().push(h.access.fraction)}}m}
fn run_nearest_profile(profile:Profile,output_dir:&Path,hospital_snaps:&[HospitalSnap],grid_snaps:&[GridSnap],grids:&[GridPoint],nodes:&[Node],segments:&[RawSegment],rev_offsets:&[usize],rev_edges:&[AdjEdge],undirected:bool)->Result<(String,u64,f64),Box<dyn std::error::Error>>{
    let start=Instant::now();let path=output_dir.join(format!("nearest_{}.bin",profile.name()));let mut dist=vec![INF;nodes.len()];let mut heap=BinaryHeap::new();
    for h in hospital_snaps{if h.access.kind==SNAP_NODE{if 0<dist[h.access.node_id as usize]{dist[h.access.node_id as usize]=0;heap.push((Reverse(0),h.access.node_id));}}else{let s=&segments[h.access.segment_id as usize];let w=profile.segment_weight(s,nodes);for (n,c) in target_connectors(h.access,s,w,undirected){if c<dist[n as usize]{dist[n as usize]=c;heap.push((Reverse(c),n));}}}}
    while let Some((Reverse(c),n))=heap.pop(){if c!=dist[n as usize]{continue}for &e in &rev_edges[rev_offsets[n as usize]..rev_offsets[n as usize+1]]{let z=c.saturating_add(profile.weight(e));if z<dist[e.to as usize]{dist[e.to as usize]=z;heap.push((Reverse(z),e.to));}}}
    let same=build_hospitals_by_segment(hospital_snaps);let mut wtr=BufWriter::new(File::create(&path)?);let mut reachable=0u64;
    for (g,gs) in grids.iter().zip(grid_snaps.iter()){let mut best=INF;if gs.access.kind==SNAP_NODE{best=dist[gs.access.node_id as usize]}else{let s=&segments[gs.access.segment_id as usize];let sw=profile.segment_weight(s,nodes);for (n,c) in origin_connectors(gs.access,s,sw,undirected){let d=dist[n as usize];if d!=INF{best=best.min(c.saturating_add(d));}}if let Some(fracs)=same.get(&gs.access.segment_id){for &hf in fracs{if let Some(c)=direct_same_segment_cost(gs.access.fraction,hf,s,sw,undirected){best=best.min(c)}}}}
        let min=if best==INF{f32::NAN}else{reachable+=1;(best as f64/DIST_MULTIPLICATOR as f64*60.0) as f32};wtr.write_all(&g.grid_id.to_le_bytes())?;wtr.write_all(&min.to_le_bytes())?;}
    wtr.flush()?;Ok((path.to_string_lossy().to_string(),reachable,start.elapsed().as_secs_f64()))
}

fn run_profile(profile:Profile,output_dir:&Path,hospital_snaps:&[HospitalSnap],grid_snaps:&[GridSnap],grids:&[GridPoint],nodes:&[Node],segments:&[RawSegment],components_with_grid:&HashSet<u32>,cutoff:u32,offsets:&[usize],edges:&[AdjEdge],rev_offsets:&[usize],rev_edges:&[AdjEdge],undirected:bool)->Result<ProfileSummary,Box<dyn std::error::Error>>{
    let start=Instant::now();let out=output_dir.join(format!("travel_{}.bin",profile.name()));let writer=Arc::new(Mutex::new(BufWriter::new(File::create(&out)?)));let (to,targets,so,sgi,_)=build_target_index(profile,nodes,segments,grid_snaps,nodes.len(),undirected);let searches=AtomicU64::new(0);let skipped=AtomicU64::new(0);let records=AtomicU64::new(0);
    hospital_snaps.par_iter().for_each_init(||Workspace::new(nodes.len()),|ws,h|{if !components_with_grid.contains(&h.access.component_id){skipped.fetch_add(1,Ordering::Relaxed);return;}searches.fetch_add(1,Ordering::Relaxed);let rows=route_one(ws,h,profile,cutoff,nodes,segments,offsets,edges,&to,&targets,&so,&sgi,grid_snaps,grids,undirected);records.fetch_add(rows.len() as u64,Ordering::Relaxed);write_travel_records(&writer,&rows).expect("write travel");});writer.lock().unwrap().flush()?;let routing_seconds=start.elapsed().as_secs_f64();let (nearest_output_file,nearest_reachable_grids,nearest_seconds)=run_nearest_profile(profile,output_dir,hospital_snaps,grid_snaps,grids,nodes,segments,rev_offsets,rev_edges,undirected)?;
    Ok(ProfileSummary{profile:profile.name().to_string(),output_file:out.to_string_lossy().to_string(),nearest_output_file,route_searches:searches.load(Ordering::Relaxed),skipped_hospitals_no_target_component:skipped.load(Ordering::Relaxed),travel_records:records.load(Ordering::Relaxed),nearest_reachable_grids,routing_seconds,nearest_seconds})
}

fn main()->Result<(),Box<dyn std::error::Error>>{
    let args:Vec<String>=env::args().collect();
    if args.len()==2 && args[1]=="--protocol-version" { println!("{}", ROUTER_PROTOCOL_VERSION); return Ok(()); }
    if args.len()!=16{usage()}
    let pbf=PathBuf::from(&args[1]);let grids_path=PathBuf::from(&args[2]);let hospitals_path=PathBuf::from(&args[3]);let out=PathBuf::from(&args[4]);let cutoff_min:f64=args[5].parse()?;let max_speed:u32=args[6].parse()?;let undirected:bool=args[7].parse()?;let threads:usize=args[8].parse()?;let profiles=parse_profiles(&args[9]);
    let component_rescue_enabled:bool=args[10].parse()?;let tiny_component_max_road_km:f64=args[11].parse()?;let component_rescue_min_size_ratio:f64=args[12].parse()?;let component_rescue_min_grid_count:u64=args[13].parse()?;let component_rescue_min_grid_ratio:f64=args[14].parse()?;let component_rescue_max_extra_km:f64=args[15].parse()?;
    if tiny_component_max_road_km<=0.0{return Err("tiny_component_max_road_km must be > 0".into())}if component_rescue_min_size_ratio<=1.0{return Err("rescue_min_size_ratio must be > 1".into())}if component_rescue_min_grid_count==0{return Err("rescue_min_grid_count must be > 0".into())}if component_rescue_min_grid_ratio<=1.0{return Err("rescue_min_grid_ratio must be > 1".into())}if component_rescue_max_extra_km<0.0{return Err("rescue_max_extra_km must be >= 0".into())}
    fs::create_dir_all(&out)?;rayon::ThreadPoolBuilder::new().num_threads(threads).build_global()?;
    println!("[1/6] Reading points");let grids=read_grid_points(&grids_path)?;let hospitals=read_hospital_points(&hospitals_path)?;println!("  grids={}, hospitals={}",grids.len(),hospitals.len());
    println!("[2/6] Parsing OSM PBF");let t=Instant::now();let (nodes,segments,road_stats,raw_max,cap_l,cap_n)=parse_pbf(&pbf,max_speed)?;let parse_seconds=t.elapsed().as_secs_f64();println!("  nodes={}, segments={}, {:.2}s",nodes.len(),segments.len(),parse_seconds);if nodes.is_empty()||segments.is_empty(){return Err("road graph is empty".into())}
    println!("[3/6] Building graph/components");let t=Instant::now();let (offsets,edges,edge_count)=build_graph(&nodes,&segments,undirected);let graph_seconds=t.elapsed().as_secs_f64();let t=Instant::now();let components=compute_components(nodes.len(),&segments);let mut component_stats=compute_component_stats(&nodes,&segments,&components);let component_seconds=t.elapsed().as_secs_f64();let (rev_offsets,rev_edges)=if undirected{(offsets.clone(),edges.clone())}else{build_reverse_graph(nodes.len(),&offsets,&edges)};
    println!("[4/6] Snapping to street network (no snap gate; two-pass component rescue={})",component_rescue_enabled);let t=Instant::now();let (grid_snaps,hospital_snaps,grid_component_rescue_count,hospital_component_rescue_count)=snap_points(&nodes,&segments,&components,&mut component_stats,&grids,&hospitals,component_rescue_enabled,tiny_component_max_road_km,component_rescue_min_size_ratio,component_rescue_min_grid_count,component_rescue_min_grid_ratio,component_rescue_max_extra_km);let snap_seconds=t.elapsed().as_secs_f64();
    let tiny_component_count=component_stats.iter().filter(|x|x.road_length_km<tiny_component_max_road_km).count();let sparse_grid_component_count=component_stats.iter().filter(|x|x.raw_grid_count<component_rescue_min_grid_count).count();let suspicious_component_count=component_stats.iter().filter(|x|x.road_length_km<tiny_component_max_road_km||x.raw_grid_count<component_rescue_min_grid_count).count();write_component_stats(&out.join("component_stats.csv"),&component_stats,tiny_component_max_road_km,component_rescue_min_grid_count)?;println!("  components={}, tiny(<{:.3} km)={}, sparse_grid(<{})={}, suspicious={}",component_stats.len(),tiny_component_max_road_km,tiny_component_count,component_rescue_min_grid_count,sparse_grid_component_count,suspicious_component_count);
    write_grid_snap(&out.join("grid_snap.bin"),&grid_snaps)?;write_hospital_snap(&out.join("hospital_snap.bin"),&hospital_snaps)?;let ge=grid_snaps.iter().filter(|x|x.access.kind==SNAP_EDGE).count();let he=hospital_snaps.iter().filter(|x|x.access.kind==SNAP_EDGE).count();println!("  grid edge/node={}/{}, rescued={}; hospital edge/node={}/{}, rescued={}",ge,grid_snaps.len()-ge,grid_component_rescue_count,he,hospital_snaps.len()-he,hospital_component_rescue_count);
    let components_with_grid: HashSet<u32> = grid_snaps.iter().map(|g| g.access.component_id).collect();
    let cutoff=((cutoff_min/60.0)*DIST_MULTIPLICATOR as f64).round() as u32;
    println!("[5/6] Routing profiles");let mut summaries=Vec::new();for p in profiles{summaries.push(run_profile(p,&out,&hospital_snaps,&grid_snaps,&grids,&nodes,&segments,&components_with_grid,cutoff,&offsets,&edges,&rev_offsets,&rev_edges,undirected)?)}
    let mut rw=BufWriter::new(File::create(out.join("road_class_stats.csv"))?);writeln!(rw,"highway,way_count,segment_count,included_way_count,included_segment_count,explicit_maxspeed_way_count,fallback_speed_way_count,legacy_default_speed_kmh,chn_osm_default_speed_kmh")?;for (h,s) in road_stats{writeln!(rw,"{},{},{},{},{},{},{},{},{}",h,s.way_count,s.segment_count,s.included_way_count,s.included_segment_count,s.explicit_maxspeed_way_count,s.fallback_speed_way_count,s.legacy_default_speed_kmh,s.chn_osm_default_speed_kmh)?;}rw.flush()?;
    println!("[6/6] Writing summary");let summary=RouterSummary{pbf:pbf.to_string_lossy().to_string(),grid_count:grids.len(),hospital_count:hospitals.len(),node_count:nodes.len(),raw_segment_count:segments.len(),directed_edge_count:edge_count,undirected,cutoff_min,max_speed_kmh:max_speed,snap_gate_applied:false,snap_semantics:"nearest legal street access: non-motorway segments use projected edge position; motorway interior snapping disabled and nearest routable node remains eligible; raw snapping is performed first for all populated grids and hospitals; components with short road length or sparse populated-grid support are suspicious, and may be rescued only to a nearby component with adequate populated-grid support and substantially greater road/grid support within the configured extra snap distance; snap distance remains QC only and there is no hard snap gate".to_string(),motorway_edge_snap_allowed:false,motorway_link_edge_snap_allowed:true,component_rescue_enabled,tiny_component_max_road_km,component_rescue_min_size_ratio,component_rescue_min_grid_count,component_rescue_min_grid_ratio,component_rescue_max_extra_km,component_count:component_stats.len(),tiny_component_count,sparse_grid_component_count,suspicious_component_count,grid_component_rescue_count,hospital_component_rescue_count,grid_edge_snap_count:ge,grid_node_snap_count:grid_snaps.len()-ge,hospital_edge_snap_count:he,hospital_node_snap_count:hospital_snaps.len()-he,parse_seconds,graph_seconds,component_seconds,snap_seconds,raw_max_speed_kmh_seen:raw_max,speed_capped_legacy_way_count:cap_l,speed_capped_chn_osm_default_way_count:cap_n,profiles:summaries};serde_json::to_writer_pretty(File::create(out.join("router_summary.json"))?,&summary)?;Ok(())
}
