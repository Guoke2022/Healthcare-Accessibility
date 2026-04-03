"""
Linux workflow for producing accessibility datasets with Ray and osm_ch_web.

This public version keeps the original core processing logic while exposing the
main runtime settings through CLI arguments so the script can be documented and
reused outside the original local environment.
"""

import argparse
import os
import ray
import math
import time
import signal
import subprocess
import pandas as pd
import geopandas as gpd
import logging
from tqdm import tqdm
import psutil
import warnings
from pathlib import Path

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
YEAR = 2018
ENABLE_SAMPLING = False
SAMPLE_FRACTION = 0.05
CALCULATE_LEVEL_3_ONLY = True

RESULTS_DIR = str(SCRIPT_DIR / "results" / str(YEAR))
TMP_DIR = str(SCRIPT_DIR / "tmp" / str(YEAR))
LOG_FILE_PATH = os.path.join(TMP_DIR, f"accessibility_{YEAR}.log")

OSM_CH_WEB_BASE_URL = "http://localhost:11114/dijkstra"
OSM_CH_WEB_START_CMD_TEMPLATE = "cargo run --release -p osm_ch_web {}"
OSM_CH_WEB_STARTUP_TIME = 30
OSM_CH_REQUEST_TIMEOUT = 0.5
OSM_CH_REQUEST_MAX_RETRIES_PER_CALL = 3
OSM_CH_SERVICE_MAX_RESTARTS_PER_REGION = 2
OSM_CH_SERVICE_RESTART_COOLDOWN = 30

# Individual processing configuration  
SINGLE_HOSPITAL_MAX_RETRY_ATTEMPTS = 3
SINGLE_GRID_MAX_RETRY_ATTEMPTS = 3

# CPU-based parallel control configuration
CPU_USAGE_THRESHOLD = 95
S1_TARGET_CPU_USAGE = 88.0
S2_TARGET_CPU_USAGE = 92.0
DEFAULT_BOUNDARIES_FILENAME = "china_boundaries.geojson"
CHINA_BOUNDARIES_GEOJSON = str(SCRIPT_DIR / DEFAULT_BOUNDARIES_FILENAME)
HOSPITALS_DATA_CSV = str(SCRIPT_DIR / f"{YEAR}.csv")

SHOW_OSMCH_SERVICE_LOG = False

# --- Logger Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def configure_logger(log_file_path):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Produce regional accessibility datasets on Linux with Ray and osm_ch_web.",
    )
    parser.add_argument("--year", type=int, default=YEAR, help="Analysis year used for default input and output paths.")
    parser.add_argument("--boundaries-geojson", type=Path, default=None, help="Path to the national boundaries GeoJSON.")
    parser.add_argument("--hospitals-csv", type=Path, default=None, help="Path to the hospital table CSV.")
    parser.add_argument("--results-dir", type=Path, default=None, help="Directory for final region-level outputs.")
    parser.add_argument("--tmp-dir", type=Path, default=None, help="Directory for logs and checkpoints.")
    parser.add_argument("--enable-sampling", action="store_true", help="Enable hospital sampling for debugging runs.")
    parser.add_argument("--sample-fraction", type=float, default=SAMPLE_FRACTION, help="Sampling fraction when --enable-sampling is used.")
    parser.add_argument("--all-levels", action="store_true", help="Also compute Level 1 and Level 2 nearest travel times.")
    parser.add_argument("--osm-ch-web-base-url", default=OSM_CH_WEB_BASE_URL, help="Base URL for the osm_ch_web dijkstra endpoint.")
    parser.add_argument(
        "--osm-ch-web-start-cmd-template",
        default=OSM_CH_WEB_START_CMD_TEMPLATE,
        help="Command template used to launch osm_ch_web. Use braces as the .fmi placeholder.",
    )
    parser.add_argument("--osm-ch-web-startup-time", type=int, default=OSM_CH_WEB_STARTUP_TIME, help="Seconds to wait after starting osm_ch_web.")
    parser.add_argument("--osm-ch-request-timeout", type=float, default=OSM_CH_REQUEST_TIMEOUT, help="Per-request timeout in seconds.")
    parser.add_argument("--osm-ch-request-max-retries", type=int, default=OSM_CH_REQUEST_MAX_RETRIES_PER_CALL, help="HTTP retry count per call.")
    parser.add_argument("--osm-ch-service-max-restarts", type=int, default=OSM_CH_SERVICE_MAX_RESTARTS_PER_REGION, help="Maximum service restarts per region.")
    parser.add_argument("--osm-ch-service-restart-cooldown", type=int, default=OSM_CH_SERVICE_RESTART_COOLDOWN, help="Cooldown before restarting osm_ch_web.")
    parser.add_argument("--show-osmch-service-log", action="store_true", help="Forward osm_ch_web stdout and stderr instead of suppressing it.")
    return parser.parse_args()


def apply_runtime_args(args):
    global YEAR, ENABLE_SAMPLING, SAMPLE_FRACTION, CALCULATE_LEVEL_3_ONLY
    global RESULTS_DIR, TMP_DIR, LOG_FILE_PATH
    global CHINA_BOUNDARIES_GEOJSON, HOSPITALS_DATA_CSV
    global OSM_CH_WEB_BASE_URL, OSM_CH_WEB_START_CMD_TEMPLATE, OSM_CH_WEB_STARTUP_TIME
    global OSM_CH_REQUEST_TIMEOUT, OSM_CH_REQUEST_MAX_RETRIES_PER_CALL
    global OSM_CH_SERVICE_MAX_RESTARTS_PER_REGION, OSM_CH_SERVICE_RESTART_COOLDOWN
    global SHOW_OSMCH_SERVICE_LOG

    YEAR = args.year
    ENABLE_SAMPLING = args.enable_sampling
    SAMPLE_FRACTION = args.sample_fraction
    CALCULATE_LEVEL_3_ONLY = not args.all_levels

    RESULTS_DIR = str(args.results_dir or (SCRIPT_DIR / "results" / str(args.year)))
    TMP_DIR = str(args.tmp_dir or (SCRIPT_DIR / "tmp" / str(args.year)))
    LOG_FILE_PATH = os.path.join(TMP_DIR, f"accessibility_{args.year}.log")
    CHINA_BOUNDARIES_GEOJSON = str(args.boundaries_geojson or (SCRIPT_DIR / DEFAULT_BOUNDARIES_FILENAME))
    HOSPITALS_DATA_CSV = str(args.hospitals_csv or (SCRIPT_DIR / f"{args.year}.csv"))

    OSM_CH_WEB_BASE_URL = args.osm_ch_web_base_url
    OSM_CH_WEB_START_CMD_TEMPLATE = args.osm_ch_web_start_cmd_template
    OSM_CH_WEB_STARTUP_TIME = args.osm_ch_web_startup_time
    OSM_CH_REQUEST_TIMEOUT = args.osm_ch_request_timeout
    OSM_CH_REQUEST_MAX_RETRIES_PER_CALL = args.osm_ch_request_max_retries
    OSM_CH_SERVICE_MAX_RESTARTS_PER_REGION = args.osm_ch_service_max_restarts
    OSM_CH_SERVICE_RESTART_COOLDOWN = args.osm_ch_service_restart_cooldown
    SHOW_OSMCH_SERVICE_LOG = args.show_osmch_service_log

    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    configure_logger(LOG_FILE_PATH)


configure_logger(LOG_FILE_PATH)

# --- Global state for OSM CH Service (managed by driver process only) ---
osm_ch_service_process = None
current_region_fmi_path_for_service = None
osm_ch_service_restarts_in_current_region = 0

# --- OSM CH Service Management (called by driver process only) ---
def start_osm_ch_service(fmi_path, base_url_for_ping, startup_time):
    import requests

    global osm_ch_service_process, current_region_fmi_path_for_service
    if osm_ch_service_process and osm_ch_service_process.poll() is None:
        logger.warning("OSM CH service already running. Attempting to stop before restart.")
        stop_osm_ch_service()

    current_region_fmi_path_for_service = fmi_path
    cmd = OSM_CH_WEB_START_CMD_TEMPLATE.format(fmi_path)
    logger.info(f"Starting OSM CH service with command: {cmd}")
    try:
        stdout_opt = None if SHOW_OSMCH_SERVICE_LOG else subprocess.DEVNULL
        stderr_opt = None if SHOW_OSMCH_SERVICE_LOG else subprocess.DEVNULL
        osm_ch_service_process = subprocess.Popen(
            cmd, shell=True, preexec_fn=os.setsid,
            stdout=stdout_opt, stderr=stderr_opt
        )
        logger.info(f"OSM CH service starting (PID: {osm_ch_service_process.pid}). Waiting {startup_time}s for it to initialize...")
        time.sleep(startup_time)

        check_url = base_url_for_ping.split('/dijkstra')[0]
        if not check_url.endswith('/'): check_url += '/'
        try:
            ping_response = requests.get(check_url, timeout=2)
            if 200 <= ping_response.status_code < 300 or ping_response.status_code == 404:
                 logger.info(f"OSM CH service ping check successful (status: {ping_response.status_code}).")
            else:
                 logger.warning(f"OSM CH service ping check returned non-ok status: {ping_response.status_code}.")
        except requests.exceptions.ConnectionError:
            logger.error("OSM CH service might not be fully up after startup time (ping failed connection).")
            raise
        except requests.exceptions.Timeout:
            logger.warning("OSM CH service ping check timed out.")
        except Exception as e_ping:
            logger.warning(f"OSM CH service ping check failed with unexpected error: {e_ping}")

    except Exception as e_start:
        logger.error(f"Failed to start OSM CH service: {e_start}", exc_info=True)
        osm_ch_service_process = None
        raise

def stop_osm_ch_service():
    global osm_ch_service_process
    if osm_ch_service_process and osm_ch_service_process.pid:
        pid_to_stop = osm_ch_service_process.pid
        logger.info(f"Stopping OSM CH service (PID: {pid_to_stop}).")
        try:
            pgid = os.getpgid(pid_to_stop)
            os.killpg(pgid, signal.SIGTERM)
            osm_ch_service_process.wait(timeout=10)
            logger.info(f"OSM CH service stopped (PID: {pid_to_stop}, SIGTERM).")
        except ProcessLookupError:
            logger.warning(f"OSM CH service process (PID: {pid_to_stop}) not found during SIGTERM (already exited?).")
        except PermissionError:
             logger.warning(f"Permission error stopping OSM CH service process (PID: {pid_to_stop}) with SIGTERM.")
        except TimeoutError:
            logger.warning(f"OSM CH service (PID: {pid_to_stop}) did not terminate gracefully (SIGTERM timeout). Attempting SIGKILL.")
            try:
                os.killpg(pgid, signal.SIGKILL)
                osm_ch_service_process.wait(timeout=5)
                logger.info(f"OSM CH service stopped (PID: {pid_to_stop}, SIGKILL).")
            except ProcessLookupError:
                logger.warning(f"OSM CH service process (PID: {pid_to_stop}) not found during SIGKILL.")
            except Exception as e2:
                logger.error(f"Failed to SIGKILL OSM CH service (PID: {pid_to_stop}): {e2}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while stopping OSM CH service (PID: {pid_to_stop}): {e}")
        finally:
            osm_ch_service_process = None
    else:
        logger.info("OSM CH service not running or no PID to stop.")


def handle_osm_service_failure_for_driver(base_url_for_ping, startup_time, restart_cooldown):
    global osm_ch_service_restarts_in_current_region
    logger.warning("OSM CH service request failed at worker level. Driver attempting service restart.")
    stop_osm_ch_service()

    osm_ch_service_restarts_in_current_region += 1
    if osm_ch_service_restarts_in_current_region > OSM_CH_SERVICE_MAX_RESTARTS_PER_REGION:
        logger.error(f"Max service restarts ({OSM_CH_SERVICE_MAX_RESTARTS_PER_REGION}) reached for the current region. Aborting further requests for this region.")
        return False

    logger.info(f"Cooling down for {restart_cooldown}s before restarting service.")
    time.sleep(restart_cooldown)

    try:
        if current_region_fmi_path_for_service is None:
            logger.error("Cannot restart service: FMI path for current region is unknown.")
            return False
        start_osm_ch_service(current_region_fmi_path_for_service, base_url_for_ping, startup_time)
        return True
    except Exception as e_restart:
        logger.error(f"Failed to restart OSM CH service: {e_restart}. Aborting.", exc_info=True)
        return False

# --- Worker-level HTTP Request Function ---
def _get_travel_time_for_worker(start_coords, end_coords, base_url, timeout=0.5, max_retries=3):
    import requests
    import json
    import time

    OFFSET_DEGREES = 0.001

    start_offsets = [
        (0, 0),
        (OFFSET_DEGREES, 0),
        (-OFFSET_DEGREES, 0),
        (0, OFFSET_DEGREES),
        (0, -OFFSET_DEGREES),
    ]

    end_offsets = [
        (0, 0),
        (OFFSET_DEGREES, 0),
        (-OFFSET_DEGREES, 0),
        (0, OFFSET_DEGREES),
        (0, -OFFSET_DEGREES),
    ]

    offsets = []
    for start_offset in start_offsets:
        for end_offset in end_offsets:
            offsets.append((start_offset, end_offset))

    headers = {"Content-Type": "application/json"}
    
    for attempt, (start_offset, end_offset) in enumerate(offsets):

        adjusted_start = [start_coords[0] + start_offset[0], start_coords[1] + start_offset[1]]
        adjusted_end = [end_coords[0] + end_offset[0], end_coords[1] + end_offset[1]]
        
        payload = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": adjusted_start}},
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": adjusted_end}},
            ],
        }

        try:
            response = requests.post(base_url, data=json.dumps(payload), headers=headers, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            cost = data["features"][0]["properties"]["weight"]
            if cost == "no path found":
                if attempt < len(offsets) - 1:
                    time.sleep(0.002)
                    continue
                else:
                    return None
            return float(cost)
            
        except (requests.exceptions.Timeout, requests.exceptions.HTTPError, 
                requests.exceptions.ConnectionError, json.JSONDecodeError, Exception):
            if attempt < len(offsets) - 1:
                time.sleep(0.002)
                continue


    return None

# --- Data Validation Functions ---
def validate_r_value(r_value, hospital_id, h_beds, current_pw_sum):
    """Validate whether a hospital-level R value is numerically plausible."""
    if r_value is None:
        logger.warning(f"閸栧娅?{hospital_id} R閸婇棿璐烴one")
        return False, "R_VALUE_NONE"
    
    if not isinstance(r_value, (int, float)):
        logger.warning(f"閸栧娅?{hospital_id} R閸婅偐琚崹瀣晩鐠? {type(r_value)}")
        return False, "R_VALUE_TYPE_ERROR"
    
    if r_value < 0:
        logger.warning(f"閸栧娅?{hospital_id} R閸婇棿璐熺拹鐔告殶: {r_value}")
        return False, "R_VALUE_NEGATIVE"
    
    if r_value == 0:
        logger.warning(f"閸栧娅?{hospital_id} R閸婇棿璐?, beds={h_beds}, pw_sum={current_pw_sum}")
        return False, "R_VALUE_ZERO"
    
    if r_value > 100:
        logger.warning(f"閸栧娅?{hospital_id} R閸婅壈绻冩径? {r_value}, beds={h_beds}, pw_sum={current_pw_sum}")
        return False, "R_VALUE_TOO_LARGE"
    
    if math.isnan(r_value) or math.isinf(r_value):
        logger.warning(f"閸栧娅?{hospital_id} R閸婇棿璐烴aN閹存湏nf: {r_value}")
        return False, "R_VALUE_INVALID"
    
    return True, "VALID"

def validate_pw_sum(pw_sum, hospital_id):
    """Validate whether the weighted population sum is numerically plausible."""
    if pw_sum is None:
        return False, "PW_SUM_NONE"
    
    if not isinstance(pw_sum, (int, float)):
        return False, "PW_SUM_TYPE_ERROR"
    
    if pw_sum < 0:
        logger.warning(f"閸栧娅?{hospital_id} 娴滃搫褰涢弶鍐櫢閹鎷版稉楦跨: {pw_sum}")
        return False, "PW_SUM_NEGATIVE"
    
    if math.isnan(pw_sum) or math.isinf(pw_sum):
        logger.warning(f"閸栧娅?{hospital_id} 娴滃搫褰涢弶鍐櫢閹鎷版稉绡榓N閹存湏nf: {pw_sum}")
        return False, "PW_SUM_INVALID"
    
    return True, "VALID"

# --- CPU-based Parallel Control ---
def should_limit_parallelism():
    """
    濡偓閺屻儲妲搁崥锕傛付鐟曚線妾洪崚璺鸿嫙鐞涘苯瀹抽敍鍫濈唨娴滃钉PU娴ｈ法鏁ら悳鍥风礆
    """
    try:
        current_cpu = psutil.cpu_percent(interval=0.1)
    except Exception as e:
        logger.warning(f"閺冪姵纭堕懢宄板絿CPU娴ｈ法鏁ら悳? {e}")
        return False
def calculate_dynamic_batch_size():
    """
    閺嶈宓佺化鑽ょ埠鐠у嫭绨崝銊︹偓浣筋吀缁犳澹掑▎鈥炽亣鐏?    """
    try:

        cpu_count = psutil.cpu_count()
        memory_info = psutil.virtual_memory()
        current_cpu = psutil.cpu_percent(interval=0.1)
        

        base_batch_size = max(50, cpu_count * 20)
        

        available_memory_gb = memory_info.available / (1024**3)
        if available_memory_gb > 16:
            memory_factor = 1.5
        elif available_memory_gb > 8:
            memory_factor = 1.2
        elif available_memory_gb > 4:
            memory_factor = 1.0
        else:
            memory_factor = 0.7
        
        if current_cpu >= 98:
            cpu_factor = 0.3
        elif current_cpu >= 96:
            cpu_factor = 0.5
        elif current_cpu >= 95:
            cpu_factor = 0.7
        else:
            cpu_factor = 1.0
        

        

        dynamic_batch_size = max(10, min(dynamic_batch_size, 2000))
        
        logger.debug(f"閸斻劍鈧焦澹掑▎鈥炽亣鐏? {dynamic_batch_size} (CPU閺嶇绺?{cpu_count}, 閸欘垳鏁ら崘鍛摠:{available_memory_gb:.1f}GB, CPU娴ｈ法鏁ら悳?{current_cpu:.1f}%)")
        return dynamic_batch_size
        
    except Exception as e:
        logger.warning(f"Failed to calculate dynamic batch size: {e}. Falling back to a conservative default.")
        return max(50, psutil.cpu_count() * 10) if psutil else 200

def execute_tasks_with_cpu_control(task_creation_func, task_data, *args):
    """
    閺嶈宓丆PU娴ｈ法鏁ら悳鍥ㄦ閼宠姤澧界悰灞兼崲閸?    task_creation_func: 閸掓稑缂撴禒璇插閻ㄥ嫬鍤遍弫?    task_data: 娴犺濮熼弫鐗堝祦閸掓銆?
    *args: 娴肩娀鈧帞绮皌ask_creation_func閻ㄥ嫰顤傛径鏍у棘閺?    """
    if not task_data:
        return []
    

    if should_limit_parallelism():
        batch_size = calculate_dynamic_batch_size()
        logger.debug(f"CPU usage is high; processing tasks in batches (batch_size={batch_size}).")
        

        all_results = []
        for i in range(0, len(task_data), batch_size):
            batch = task_data[i:i + batch_size]
            batch_tasks = [task_creation_func(item, *args) for item in batch]
            batch_results = ray.get(batch_tasks)
            all_results.extend(batch_results)
        return all_results
    else:
        logger.debug(f"CPU usage is below {CPU_USAGE_THRESHOLD}%; dispatching all tasks without batching.")
        all_tasks = [task_creation_func(item, *args) for item in task_data]
        return ray.get(all_tasks)

# --- Core Calculation Logic ---
def gaussian_decay_weight(travel_time_input, threshold_minutes):
    travel_time_minutes = travel_time_input
    if travel_time_minutes is None or travel_time_minutes > threshold_minutes:
        return 0.0
    val = (math.exp(-0.5 * (travel_time_minutes / threshold_minutes) ** 2) - math.exp(-0.5)) / (1 - math.exp(-0.5))
    return max(0, val)

# --- Ray Remote Tasks ---
@ray.remote(num_cpus=0.5)
def s1_calculate_pw_for_grid(grid_centroid_xy, grid_pop_sum, hospital_coords_xy, hospital_threshold,
                             osm_ch_web_base_url_param, osm_ch_request_timeout_param, osm_ch_request_max_retries_param):
    travel_time_result = _get_travel_time_for_worker(
        hospital_coords_xy, grid_centroid_xy,
        osm_ch_web_base_url_param, osm_ch_request_timeout_param, osm_ch_request_max_retries_param
    )
    if travel_time_result is None:
        return None
    travel_time_minutes = travel_time_result
    weight = gaussian_decay_weight(travel_time_minutes, hospital_threshold)
    pw = grid_pop_sum * weight
    return pw if not math.isnan(pw) else 0.0

@ray.remote(num_cpus=0.5)
def s2_calculate_rw_for_hospital(hospital_data, grid_centroid_xy,
                                 osm_ch_web_base_url_param, osm_ch_request_timeout_param, osm_ch_request_max_retries_param):
    h_lng, h_lat, h_grade, h_R = hospital_data
    h_coords = [h_lng, h_lat]
    grade_level_numeric = 0
    if h_grade.startswith("Level3"):
        threshold = 60; grade_level_numeric = 3
    elif h_grade.startswith("Level2"):
        threshold = 45; grade_level_numeric = 2
    else:
        threshold = 30; grade_level_numeric = 1
    travel_time_result = _get_travel_time_for_worker(
        grid_centroid_xy, h_coords,
        osm_ch_web_base_url_param, osm_ch_request_timeout_param, osm_ch_request_max_retries_param
    )
    if travel_time_result is None:
        return None
    travel_time_minutes = travel_time_result
    weight = gaussian_decay_weight(travel_time_minutes, threshold)
    rw = h_R * weight
    return [rw if not math.isnan(rw) else 0.0, travel_time_minutes, grade_level_numeric]

# --- Ray Remote Tasks for Single Item Processing ---
@ray.remote(num_cpus=0.5)
def s1_process_single_hospital(hospital_info, grids_data_input, config_params, max_retry_attempts=3):
    """
    婢跺嫮鎮婇崡鏇氶嚋閸栧娅岄惃鍑撮崐鑹邦吀缁犳绱濈敮锕佸殰閸斻劑鍣哥拠鏇熸簚閸?    """
    hospital_id, h_lng, h_lat, h_beds, h_grade = hospital_info
    h_coords = [h_lng, h_lat]
    
    if pd.isna(h_beds):
        h_beds_float = 5.0
    else:
        try: 
            h_beds_float = float(h_beds)
        except ValueError: 
            h_beds_float = 5.0

    if h_grade.startswith("Level3"):
        threshold = 60
    elif h_grade.startswith("Level2"):
        threshold = 45
    else: 
        threshold = 30


    if isinstance(grids_data_input, ray.ObjectRef):
        grids_data = ray.get(grids_data_input)
    else:
        grids_data = grids_data_input

    for attempt in range(max_retry_attempts):
        try:

            valid_grids = []
            for grid_centroid_x, grid_centroid_y, grid_pop_sum, _ in grids_data:
                if grid_pop_sum > 0:
                    valid_grids.append((grid_centroid_x, grid_centroid_y, grid_pop_sum))

            if not valid_grids:
                logger.warning(f"閸栧娅?{hospital_id} 濞屸剝婀侀張澶嬫櫏閻ㄥ嫮缍夐弽鑲╁仯")
                return {'hospital_id': hospital_id, 'R': 0.0, 'status': 'NO_VALID_GRIDS'}


                grid_centroid_x, grid_centroid_y, grid_pop_sum = grid_data
                return s1_calculate_pw_for_grid.remote(
                    (grid_centroid_x, grid_centroid_y), grid_pop_sum, h_coords, threshold,
                    config_params['OSM_CH_WEB_BASE_URL'], 
                    config_params.get('OSM_CH_REQUEST_TIMEOUT', 0.5), 
                    config_params.get('OSM_CH_REQUEST_MAX_RETRIES_PER_CALL', 3)
                )

            pw_values = execute_tasks_with_cpu_control(create_grid_task, valid_grids)
            
            valid_pw_values = [val for val in pw_values if val is not None and isinstance(val, (int, float))]
            failed_grids = len(pw_values) - len(valid_pw_values)
            
            if failed_grids > 0:
                logger.debug(f"Hospital {hospital_id} had {failed_grids} grids without a valid path result.")


            if not valid_pw_values:
                logger.warning(f"閸栧娅?{hospital_id} 濞屸剝婀佹禒璁崇秿閺堝鏅ラ惃鍕壐缂冩垵褰茬拋锛勭暬")
                return {'hospital_id': hospital_id, 'R': 0.0, 'status': 'NO_VALID_PATHS'}


            current_pw_sum = sum(float(val) for val in valid_pw_values)
            

            pw_valid, pw_error = validate_pw_sum(current_pw_sum, hospital_id)
            if not pw_valid:
                if attempt < max_retry_attempts - 1:
                    logger.warning(f"閸栧娅?{hospital_id} 缁?{attempt+1} 濞嗏€崇毦鐠囨槈W閹鎷板鍌氱埗: {pw_error}閿涘矂鍣哥拠鏇氳厬...")
                    continue
                else:
                    logger.error(f"閸栧娅?{hospital_id} PW閹鎷版宀冪槈婢惰精瑙? {pw_error}")
                    return {'hospital_id': hospital_id, 'R': 0.0, 'status': f'PW_INVALID_{pw_error}'}


            

            if not r_valid:
                if attempt < max_retry_attempts - 1:
                    logger.warning(f"閸栧娅?{hospital_id} 缁?{attempt+1} 濞嗏€崇毦鐠囨槏閸婄厧绱撶敮? {r_error} (R={r_value})閿涘矂鍣哥拠鏇氳厬...")
                    continue
                else:
                    logger.error(f"閸栧娅?{hospital_id} R閸婂ジ鐛欑拠浣搞亼鐠? {r_error}")
                    return {'hospital_id': hospital_id, 'R': 0.0, 'status': f'R_INVALID_{r_error}'}


            logger.debug(f"閸栧娅?{hospital_id} 鐠侊紕鐣婚幋鎰: R={r_value:.3f}")
            return {'hospital_id': hospital_id, 'R': r_value, 'status': 'SUCCESS'}

        except Exception as e:
            if attempt < max_retry_attempts - 1:
                logger.warning(f"閸栧娅?{hospital_id} 缁?{attempt+1} 濞嗏€崇毦鐠囨洖鍤悳鏉跨磽鐢? {e}閿涘矂鍣哥拠鏇氳厬...")
                continue
            else:
                logger.error(f"閸栧娅?{hospital_id} 缂佸繗绻?{max_retry_attempts} 濞嗏€崇毦鐠囨洑绮涙径杈Е: {e}")
                return {'hospital_id': hospital_id, 'R': "EXCEPTION_FINAL", 'status': 'FAILED'}

    return {'hospital_id': hospital_id, 'R': "MAX_RETRIES_EXCEEDED", 'status': 'FAILED'}

@ray.remote(num_cpus=0.5)
def s2_process_single_grid(grid_info, hospitals_data_input, config_params, max_retry_attempts=3):
    """
    婢跺嫮鎮婇崡鏇氶嚋閺嶈偐缍夐惃鍕讲鏉堢偓鈧嗩吀缁犳绱濈敮锕佸殰閸斻劑鍣哥拠鏇熸簚閸?    """
    grid_id, g_centroid_x, g_centroid_y = grid_info
    g_coords = [g_centroid_x, g_centroid_y]
    

    if isinstance(hospitals_data_input, ray.ObjectRef):
        hospitals_data = ray.get(hospitals_data_input)
    else:
        hospitals_data = hospitals_data_input

    for attempt in range(max_retry_attempts):
        try:
            acc_sum = 0.0
            nearest_L3_time, nearest_L2_time, nearest_L1_time = -1.0, -1.0, -1.0
            
            if not hospitals_data:
                logger.warning(f"Grid {grid_id} has no valid hospitals for S2 processing.")
                return {'grid_id': grid_id, 'acc': 0.0, 'nearest_L3_time': -1.0, 
                       'nearest_L2_time': -1.0, 'nearest_L1_time': -1.0, 'status': 'NO_VALID_HOSPITALS'}

            def create_hospital_task(h_info):
                return s2_calculate_rw_for_hospital.remote(
                    h_info, g_coords,
                    config_params['OSM_CH_WEB_BASE_URL'],
                    config_params.get('OSM_CH_REQUEST_TIMEOUT', 0.5),
                    config_params.get('OSM_CH_REQUEST_MAX_RETRIES_PER_CALL', 3)
                )

            hospital_contributions = execute_tasks_with_cpu_control(create_hospital_task, hospitals_data)
            

            valid_contributions = []
            failed_hospitals = 0
            
            for contrib in hospital_contributions:
                if contrib is not None and len(contrib) == 3:
                    rw, cost, grade_level = contrib

                    if isinstance(rw, (int, float)):
                        valid_contributions.append(contrib)
                    else:
                        failed_hospitals += 1
                else:
                    failed_hospitals += 1
            
            if failed_hospitals > 0:
                logger.debug(f"Grid {grid_id} had {failed_hospitals} hospitals without a valid path result.")

            for rw, cost, grade_level in valid_contributions:
                if isinstance(rw, (int, float)):
                    acc_sum += rw
                
                if cost != -1.0:
                    if grade_level == 3 and (nearest_L3_time == -1.0 or cost < nearest_L3_time):
                        nearest_L3_time = cost
                    elif grade_level == 2 and (nearest_L2_time == -1.0 or cost < nearest_L2_time):
                        nearest_L2_time = cost
                    elif grade_level == 1 and (nearest_L1_time == -1.0 or cost < nearest_L1_time):
                        nearest_L1_time = cost


            if acc_sum < 0 or math.isnan(acc_sum) or math.isinf(acc_sum):
                if attempt < max_retry_attempts - 1:
                    logger.warning(f"閺嶈偐缍?{grid_id} 缁?{attempt+1} 濞嗏€崇毦鐠囨洖褰叉潏鐐偓褍鈧厧绱撶敮? {acc_sum}閿涘矂鍣哥拠鏇氳厬...")
                    continue
                else:
                    logger.error(f"閺嶈偐缍?{grid_id} 閸欘垵鎻幀褍鈧ジ鐛欑拠浣搞亼鐠? {acc_sum}")
                    acc_sum = 0.0


            logger.debug(f"閺嶈偐缍?{grid_id} 鐠侊紕鐣婚幋鎰: acc={acc_sum:.3f}")
            return {
                'grid_id': grid_id, 
                'acc': acc_sum,
                'nearest_L3_time': nearest_L3_time, 
                'nearest_L2_time': nearest_L2_time, 
                'nearest_L1_time': nearest_L1_time,
                'status': 'SUCCESS'
            }

        except Exception as e:
            if attempt < max_retry_attempts - 1:
                logger.warning(f"閺嶈偐缍?{grid_id} 缁?{attempt+1} 濞嗏€崇毦鐠囨洖鍤悳鏉跨磽鐢? {e}閿涘矂鍣哥拠鏇氳厬...")
                continue
            else:
                logger.error(f"閺嶈偐缍?{grid_id} 缂佸繗绻?{max_retry_attempts} 濞嗏€崇毦鐠囨洑绮涙径杈Е: {e}")
                return {
                    'grid_id': grid_id, 
                    'acc': "EXCEPTION_FINAL", 
                    'nearest_L3_time': "FAIL", 
                    'nearest_L2_time': "FAIL", 
                    'nearest_L1_time': "FAIL",
                    'status': 'FAILED'
                }

    return {
        'grid_id': grid_id, 
        'acc': "MAX_RETRIES_EXCEEDED", 
        'nearest_L3_time': "FAIL", 
        'nearest_L2_time': "FAIL", 
        'nearest_L1_time': "FAIL",
        'status': 'FAILED'
    }

# --- Main Region Processing Function ---
def process_region(region_name_cn, region_geo_df, all_hospitals_master_gdf, config_params):
    global osm_ch_service_restarts_in_current_region
    osm_ch_service_restarts_in_current_region = 0

    region_id_for_filenames = region_name_cn
    region_pop_grid_geojson = os.path.join(config_params['TMP_DIR'], f"{region_id_for_filenames}.geojson")
    region_roads_fmi = os.path.join(config_params['TMP_DIR'], f"{region_id_for_filenames}_roads.osm.pbf.fmi")

    if not os.path.exists(region_pop_grid_geojson):
        logger.error(f"Population grid GeoJSON {region_pop_grid_geojson} not found for {region_name_cn}. Skipping region.")
        return
    if not os.path.exists(region_roads_fmi):
        logger.error(f"Roads FMI file {region_roads_fmi} not found for {region_name_cn}. Skipping region.")
        return

    logger.info(f"Loading population grid data for {region_name_cn} from {region_pop_grid_geojson}")
    try:
        grids_in_region_gdf = gpd.read_file(region_pop_grid_geojson)
    except Exception as e:
        logger.error(f"Failed to load grid GeoJSON for {region_name_cn}: {e}", exc_info=True); return

    if grids_in_region_gdf.empty:
        logger.info(f"No grid cells found in {region_pop_grid_geojson} for {region_name_cn}. Skipping."); return
    if 'grid_id' not in grids_in_region_gdf.columns: grids_in_region_gdf['grid_id'] = grids_in_region_gdf.index
    if 'sum' not in grids_in_region_gdf.columns:
        logger.warning(f"'sum' column (population) not found in grids for {region_name_cn}. Setting to 0.")
        grids_in_region_gdf['sum'] = 0
    else: grids_in_region_gdf['sum'] = pd.to_numeric(grids_in_region_gdf['sum'], errors='coerce').fillna(0)

    logger.info(f"Clipping hospitals for {region_name_cn}.")
    hospitals_in_region_gdf = gpd.clip(all_hospitals_master_gdf.copy(), region_geo_df)
    if 'hospital_id' not in hospitals_in_region_gdf.columns: hospitals_in_region_gdf['hospital_id'] = hospitals_in_region_gdf.index

    output_gpkg_path_empty = os.path.join(config_params['RESULTS_DIR'], f"{region_id_for_filenames}_acc.gpkg")
    if hospitals_in_region_gdf.empty:
        logger.info(f"No hospitals found in {region_name_cn} after clipping. Saving empty results to GPKG.")
        grids_in_region_gdf['acc'] = 0.0; grids_in_region_gdf['nearest_L3_time'] = -1.0
        cols_to_save = ['grid_id', 'geometry', 'sum', 'acc', 'nearest_L3_time']
        final_cols_present = [col for col in cols_to_save if col in grids_in_region_gdf.columns]
        if 'geometry' in final_cols_present : grids_in_region_gdf[final_cols_present].to_file(output_gpkg_path_empty, driver="GPKG")
        else: logger.warning(f"Cannot save empty GPKG for {region_name_cn} as geometry column is missing from grids_in_region_gdf."); return

    if config_params['ENABLE_SAMPLING'] and len(hospitals_in_region_gdf) * config_params['SAMPLE_FRACTION'] >= 1:
        hospitals_in_region_gdf = hospitals_in_region_gdf.sample(frac=config_params['SAMPLE_FRACTION'], random_state=42)
        logger.info(f"Sampled {len(hospitals_in_region_gdf)} hospitals for {region_name_cn}.")

    if config_params['CALCULATE_LEVEL_3_ONLY']:
        original_count = len(hospitals_in_region_gdf)
        hospitals_in_region_gdf = hospitals_in_region_gdf[
            hospitals_in_region_gdf["grade"].astype(str).str.startswith("Level3")
        ]
        logger.info(f"Filtered to {len(hospitals_in_region_gdf)} Level 3 hospitals (from {original_count}) for {region_name_cn}.")
        if hospitals_in_region_gdf.empty:
            logger.info(f"No Level 3 hospitals in {region_name_cn} after filtering. Saving empty results to GPKG.")
            grids_in_region_gdf['acc'] = 0.0; grids_in_region_gdf['nearest_L3_time'] = -1.0
            cols_to_save = ['grid_id', 'geometry', 'sum', 'acc', 'nearest_L3_time']
            final_cols_present = [col for col in cols_to_save if col in grids_in_region_gdf.columns]
            if 'geometry' in final_cols_present : grids_in_region_gdf[final_cols_present].to_file(output_gpkg_path_empty, driver="GPKG")
            else: logger.warning(f"Cannot save empty GPKG for {region_name_cn} (no L3 hospitals) as geometry column is missing."); return

    try:
        start_osm_ch_service(region_roads_fmi, config_params['OSM_CH_WEB_BASE_URL'], config_params['OSM_CH_WEB_STARTUP_TIME'])
    except Exception:
        logger.error(f"Failed to start OSM service for {region_name_cn}. Skipping region."); return

    s1_checkpoint_path = os.path.join(config_params['TMP_DIR'], f"{region_id_for_filenames}_S1_hospitals_R.csv")
    if 'R' not in hospitals_in_region_gdf.columns: hospitals_in_region_gdf['R'] = pd.NA
    if os.path.exists(s1_checkpoint_path):
        logger.info(f"Loading S1 checkpoint for {region_name_cn} from {s1_checkpoint_path}")
        try:
            s1_progress_df = pd.read_csv(s1_checkpoint_path, dtype={'hospital_id': hospitals_in_region_gdf['hospital_id'].dtype})
            if not s1_progress_df.empty and 'hospital_id' in s1_progress_df.columns and 'R' in s1_progress_df.columns:
                hospitals_in_region_gdf = hospitals_in_region_gdf.set_index('hospital_id')
                s1_progress_df = s1_progress_df.set_index('hospital_id')
                hospitals_in_region_gdf.update(s1_progress_df[['R']])
                hospitals_in_region_gdf = hospitals_in_region_gdf.reset_index()
                hospitals_in_region_gdf['R'] = pd.to_numeric(hospitals_in_region_gdf['R'], errors='coerce')
            else: logger.warning(f"S1 checkpoint {s1_checkpoint_path} is empty or missing required columns.")
        except Exception as e_chkpt: logger.warning(f"Could not load or merge S1 checkpoint {s1_checkpoint_path}: {e_chkpt}.")
    hospitals_to_process_s1_df = hospitals_in_region_gdf[hospitals_in_region_gdf['R'].isna()].copy()
    s1_completed_normally = True
    if not hospitals_to_process_s1_df.empty:
        logger.info(f"Step 1 for {region_name_cn}: Calculating R_j for {len(hospitals_to_process_s1_df)} hospitals.")
        

        simple_grids_data = []
        for g in grids_in_region_gdf.itertuples():
            if g.geometry and hasattr(g.geometry, 'centroid'): 
                simple_grids_data.append((g.geometry.centroid.x, g.geometry.centroid.y, g.sum, g.grid_id))
            elif g.geometry and hasattr(g.geometry, 'x'): 
                simple_grids_data.append((g.geometry.x, g.geometry.y, g.sum, g.grid_id))
        simple_grids_data_ref = ray.put(simple_grids_data)
        

        hospital_data_for_ray = []
        for h in hospitals_to_process_s1_df.itertuples():
            if h.geometry and hasattr(h.geometry, 'x'): 
                hospital_data_for_ray.append((h.hospital_id, h.geometry.x, h.geometry.y, h.beds, str(h.grade)))


        
        processed_count = 0
        failed_count = 0
        service_restart_needed = False
        
        with tqdm(total=len(hospital_data_for_ray), desc=f"S1 Processing Hospitals ({region_name_cn})") as pbar:
            for i, hospital_info in enumerate(hospital_data_for_ray):
                try:

                    
                    hospital_id = result['hospital_id']
                    status = result['status']
                    
                    if status == 'SUCCESS':

                        processed_count += 1
                    else:

                        hospitals_in_region_gdf.loc[hospitals_in_region_gdf['hospital_id'] == hospital_id, 'R'] = pd.NA
                        failed_count += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Success': processed_count, 
                        'Failed': failed_count,
                        'CPU': f"{psutil.cpu_percent():.0f}%"
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing hospital {hospital_info[0]}: {e}")
                    failed_count += 1
                    pbar.update(1)


                if (processed_count + failed_count) % 10 == 0:
                    hospitals_with_valid_r = hospitals_in_region_gdf[
                        hospitals_in_region_gdf['R'].notna() & 
                        (hospitals_in_region_gdf['R'] != "") &
                        pd.to_numeric(hospitals_in_region_gdf['R'], errors='coerce').notna()
                    ].copy()
                    if not hospitals_with_valid_r.empty:
                        hospitals_with_valid_r[['hospital_id', 'R']].to_csv(s1_checkpoint_path, index=False)
                        logger.info(f"S1 Checkpoint saved: {len(hospitals_with_valid_r)} hospitals saved")

        hospitals_with_valid_r_final = hospitals_in_region_gdf[
            hospitals_in_region_gdf['R'].notna() & 
            (hospitals_in_region_gdf['R'] != "") &
            pd.to_numeric(hospitals_in_region_gdf['R'], errors='coerce').notna()
        ].copy()
        if not hospitals_with_valid_r_final.empty:
            hospitals_with_valid_r_final[['hospital_id', 'R']].to_csv(s1_checkpoint_path, index=False)
            logger.info(f"S1 Final checkpoint saved: {len(hospitals_with_valid_r_final)} hospitals with valid R values")
        if service_restart_needed and failed_count > 0:
            logger.warning(f"S1 completed with {failed_count} failed hospitals; attempting a service restart before recovery.")
            if handle_osm_service_failure_for_driver(
                config_params["OSM_CH_WEB_BASE_URL"],
                config_params["OSM_CH_WEB_STARTUP_TIME"],
                config_params["OSM_CH_SERVICE_RESTART_COOLDOWN"],
            ):
                logger.info("Service restart succeeded; failed hospitals will be retried in the next recovery step.")
            else:
                logger.error("Service restart failed.")
                s1_completed_normally = False
                s1_completed_normally = False


        if 'simple_grids_data_ref' in locals() and simple_grids_data_ref is not None:
            try: 
                ray.experimental.delete_object_refs([simple_grids_data_ref], True)
            except Exception: 
                pass

        if not s1_completed_normally:
            stop_osm_ch_service()
            logger.error(f"S1 for {region_name_cn} aborted.")
            return
            
        logger.info(f"S1 for {region_name_cn} completed: {processed_count} success, {failed_count} failed")

    else:
        logger.info(f"Step 1 for {region_name_cn}: All hospital R_j values already available or no hospitals to process.")

    # --- Retry Failed Hospitals ---
    failed_hospitals_df = hospitals_in_region_gdf[hospitals_in_region_gdf['R'].isna()].copy()
    if not failed_hospitals_df.empty:
        logger.info(f"Found {len(failed_hospitals_df)} hospitals with failed R calculations. Attempting individual retries.")

        # Prepare grids data for retry function
        simple_grids_data_for_retry = []
        for g in grids_in_region_gdf.itertuples():
            if g.geometry and hasattr(g.geometry, 'centroid'):
                simple_grids_data_for_retry.append((g.geometry.centroid.x, g.geometry.centroid.y, g.sum, g.grid_id))
            elif g.geometry and hasattr(g.geometry, 'x'):
                simple_grids_data_for_retry.append((g.geometry.x, g.geometry.y, g.sum, g.grid_id))

        retry_success_count = 0
        for h in failed_hospitals_df.itertuples():
            if h.geometry and hasattr(h.geometry, 'x'):
                hospital_info = (h.hospital_id, h.geometry.x, h.geometry.y, h.beds, str(h.grade))
                retry_result = ray.get(s1_process_single_hospital.remote(hospital_info, simple_grids_data_for_retry, config_params))

                if retry_result['status'] == 'SUCCESS':
                    # Update the hospital's R value
                    hospitals_in_region_gdf.loc[hospitals_in_region_gdf['hospital_id'] == h.hospital_id, 'R'] = retry_result['R']
                    retry_success_count += 1
                    logger.info(f"Successfully retried hospital {h.hospital_id}: R = {retry_result['R']}")
                else:
                    logger.warning(f"Failed to retry hospital {h.hospital_id} after all attempts: {retry_result['status']}")

        logger.info(f"Retry phase completed: {retry_success_count}/{len(failed_hospitals_df)} hospitals successfully retried")

        # Save updated checkpoint with retry results (only valid R values)
        if 'hospital_id' in hospitals_in_region_gdf.columns and 'R' in hospitals_in_region_gdf.columns:
            hospitals_with_valid_r_final = hospitals_in_region_gdf[hospitals_in_region_gdf['R'].notna() & (hospitals_in_region_gdf['R'] != "")].copy()
            if not hospitals_with_valid_r_final.empty:
                hospitals_with_valid_r_final[['hospital_id', 'R']].to_csv(s1_checkpoint_path, index=False)
                logger.info(f"Updated S1 checkpoint saved with retry results: {len(hospitals_with_valid_r_final)} hospitals with valid R values")
            else:
                logger.warning(f"No hospitals with valid R values to save in final checkpoint")
    else:
        logger.info(f"No failed hospitals found for retry in {region_name_cn}")

    # --- S2 Processing ---
    s2_checkpoint_path = os.path.join(config_params['TMP_DIR'], f"{region_id_for_filenames}_S2_grids_A.csv")
    for col in ['acc', 'nearest_L3_time', 'nearest_L2_time', 'nearest_L1_time']:
        if col not in grids_in_region_gdf.columns:
            grids_in_region_gdf[col] = pd.NA if col != 'acc' else 0.0
    if os.path.exists(s2_checkpoint_path):
        logger.info(f"Loading S2 checkpoint for {region_name_cn} from {s2_checkpoint_path}")
        try:
            s2_progress_df = pd.read_csv(s2_checkpoint_path, dtype={'grid_id': grids_in_region_gdf['grid_id'].dtype})
            if not s2_progress_df.empty and 'grid_id' in s2_progress_df.columns:
                grids_in_region_gdf = grids_in_region_gdf.set_index('grid_id')
                s2_progress_df = s2_progress_df.set_index('grid_id')
                cols_to_update_s2 = [col for col in ['acc', 'nearest_L3_time', 'nearest_L2_time', 'nearest_L1_time'] if col in s2_progress_df.columns]
                if cols_to_update_s2: grids_in_region_gdf.update(s2_progress_df[cols_to_update_s2])
                grids_in_region_gdf = grids_in_region_gdf.reset_index()
                for col in cols_to_update_s2: grids_in_region_gdf[col] = pd.to_numeric(grids_in_region_gdf[col], errors='coerce')
            else: logger.warning(f"S2 checkpoint {s2_checkpoint_path} is empty or missing 'grid_id'.")
            grids_in_region_gdf['acc'] = grids_in_region_gdf['acc'].fillna(0.0)
            for col_t in ['nearest_L3_time', 'nearest_L2_time', 'nearest_L1_time']: grids_in_region_gdf[col_t] = grids_in_region_gdf[col_t].fillna(-1.0)
        except Exception as e_chkpt_s2: logger.warning(f"Could not load or merge S2 checkpoint {s2_checkpoint_path}: {e_chkpt_s2}.")

    hospitals_for_s2_df = hospitals_in_region_gdf[hospitals_in_region_gdf['R'].notna() & (hospitals_in_region_gdf['R'] > 0)].copy()
    s2_completed_normally = True
    if hospitals_for_s2_df.empty:
        logger.info(f"No serviceable hospitals (R > 0) for S2 in {region_name_cn}. Grids will retain checkpointed or default values.")
        grids_in_region_gdf['acc'] = grids_in_region_gdf['acc'].fillna(0.0)
        for col_t in ['nearest_L3_time', 'nearest_L2_time', 'nearest_L1_time']: grids_in_region_gdf[col_t] = grids_in_region_gdf[col_t].fillna(-1.0)
    else:
        grids_to_process_s2_df = grids_in_region_gdf[grids_in_region_gdf['acc'].isna() | (grids_in_region_gdf['acc'] == 0.0)].copy()
        if grids_to_process_s2_df.empty:
            logger.info(f"Step 2 for {region_name_cn}: All grid accessibility values appear to be processed or loaded from checkpoint with non-default values.")
        else:
            logger.info(f"Step 2 for {region_name_cn}: Calculating A_k for {len(grids_to_process_s2_df)} grids.")
            

            simple_hospitals_data_s2 = []
            for h in hospitals_for_s2_df.itertuples():
                if h.geometry and hasattr(h.geometry, 'x'): 
                    simple_hospitals_data_s2.append((h.geometry.x, h.geometry.y, str(h.grade), h.R))
            simple_hospitals_data_s2_ref = ray.put(simple_hospitals_data_s2)
            

            grid_data_for_ray_s2 = []
            for g in grids_to_process_s2_df.itertuples():
                if g.geometry and hasattr(g.geometry, 'centroid'): 
                    grid_data_for_ray_s2.append((g.grid_id, g.geometry.centroid.x, g.geometry.centroid.y))
                elif g.geometry and hasattr(g.geometry, 'x'): 
                    grid_data_for_ray_s2.append((g.grid_id, g.geometry.x, g.geometry.y))

            logger.info(f"Processing {len(grid_data_for_ray_s2)} grids with adaptive batch processing...")
            
            processed_count = 0
            failed_count = 0
            service_restart_needed = False
            
            target_cpu_usage = S2_TARGET_CPU_USAGE
            initial_batch_size = 50
            min_batch_size = 100
            max_batch_size = 3000
            current_batch_size = initial_batch_size
            
            cpu_adjust_threshold = 5.0
            consecutive_adjustments = 0
            max_consecutive_adjustments = 3
            save_interval = 500
            last_saved_count = 0
            
            with tqdm(total=len(grid_data_for_ray_s2), desc=f"S2 Processing Grids ({region_name_cn})") as pbar:
                grid_idx = 0
                while grid_idx < len(grid_data_for_ray_s2):

                    current_batch = grid_data_for_ray_s2[grid_idx:grid_idx + actual_batch_size]
                    
                    batch_start_time = time.time()
                    cpu_before = psutil.cpu_percent(interval=0.1)
                    cpu_per_core_before = psutil.cpu_percent(interval=0.1, percpu=True)
                    
                    try:
                        batch_tasks = [
                            s2_process_single_grid.remote(grid_info, simple_hospitals_data_s2_ref, config_params)
                            for grid_info in current_batch
                        ]
                        batch_results = ray.get(batch_tasks)
                        

                        batch_success = 0
                        batch_failed = 0
                        for result in batch_results:
                            grid_id = result['grid_id']
                            status = result['status']
                            
                            if status == 'SUCCESS':

                                grids_in_region_gdf.loc[grids_in_region_gdf['grid_id'] == grid_id, 'nearest_L3_time'] = result['nearest_L3_time']
                                if not config_params['CALCULATE_LEVEL_3_ONLY']:
                                    grids_in_region_gdf.loc[grids_in_region_gdf['grid_id'] == grid_id, 'nearest_L2_time'] = result['nearest_L2_time']
                                    grids_in_region_gdf.loc[grids_in_region_gdf['grid_id'] == grid_id, 'nearest_L1_time'] = result['nearest_L1_time']
                                batch_success += 1
                            else:

                                grids_in_region_gdf.loc[grids_in_region_gdf['grid_id'] == grid_id, 'nearest_L3_time'] = -1.0
                                if not config_params['CALCULATE_LEVEL_3_ONLY']:
                                    grids_in_region_gdf.loc[grids_in_region_gdf['grid_id'] == grid_id, 'nearest_L2_time'] = -1.0
                                    grids_in_region_gdf.loc[grids_in_region_gdf['grid_id'] == grid_id, 'nearest_L1_time'] = -1.0
                                batch_failed += 1
                        
                        processed_count += batch_success
                        failed_count += batch_failed
                        

                        cpu_after = psutil.cpu_percent(interval=0.1)
                        cpu_per_core_after = psutil.cpu_percent(interval=0.1, percpu=True)
                        

                        avg_cpu = (cpu_before + cpu_after) / 2
                        max_cpu_core = max(max(cpu_per_core_before), max(cpu_per_core_after))
                        active_cores = sum(1 for cpu in cpu_per_core_after if cpu > 50)
                        total_cores = len(cpu_per_core_after)
                        batch_duration = batch_end_time - batch_start_time
                        throughput = actual_batch_size / batch_duration if batch_duration > 0 else 0
                        

                        old_batch_size = current_batch_size

                        effective_cpu_usage = (avg_cpu + max_cpu_core) / 2
                        core_utilization_ratio = active_cores / total_cores
                        
                        if effective_cpu_usage < target_cpu_usage - cpu_adjust_threshold and core_utilization_ratio < 0.8:

                            if consecutive_adjustments < max_consecutive_adjustments:
                                current_batch_size = min(max_batch_size, int(current_batch_size * 1.3))
                                consecutive_adjustments += 1
                            else:
                                consecutive_adjustments = 0
                        elif effective_cpu_usage > target_cpu_usage + cpu_adjust_threshold or core_utilization_ratio > 0.95:

                            if consecutive_adjustments < max_consecutive_adjustments:
                                current_batch_size = max(min_batch_size, int(current_batch_size * 0.8))
                                consecutive_adjustments += 1
                            else:
                                consecutive_adjustments = 0
                        else:

                            consecutive_adjustments = 0
                        

                        pbar.set_postfix({
                            'Success': processed_count, 
                            'Failed': failed_count,
                            'CPU': f"{avg_cpu:.1f}%",
                            'MaxCore': f"{max_cpu_core:.1f}%", 
                            'ActiveCores': f"{active_cores}/{total_cores}",
                            'BatchSize': current_batch_size,
                            'Throughput': f"{throughput:.1f}/s"
                        })
                        

                        if old_batch_size != current_batch_size:
                            logger.debug(f"S2 閹佃顐兼径褍鐨拫鍐╂殻: {old_batch_size} -> {current_batch_size} (CPU: {avg_cpu:.1f}%, MaxCore: {max_cpu_core:.1f}%, ActiveCores: {active_cores}/{total_cores})")
                        
                    except Exception as e:
                        logger.error(f"Error processing batch starting at grid {grid_idx}: {e}")
                        failed_count += actual_batch_size
                        pbar.update(actual_batch_size)
                    

                    

                    current_total = processed_count + failed_count
                    if current_total >= last_saved_count + save_interval and current_total > 0:
                        if 'grid_id' in grids_in_region_gdf.columns:
                            grids_in_region_gdf[['grid_id', 'acc', 'nearest_L3_time', 'nearest_L2_time', 'nearest_L1_time']].to_csv(s2_checkpoint_path, index=False)
                            logger.debug(f"S2 Checkpoint saved: {current_total} grids processed (saved at {current_total} grids)")
                            last_saved_count = current_total
            if 'grid_id' in grids_in_region_gdf.columns:
                grids_in_region_gdf[['grid_id', 'acc', 'nearest_L3_time', 'nearest_L2_time', 'nearest_L1_time']].to_csv(s2_checkpoint_path, index=False)
                logger.debug(f"S2 Final checkpoint saved: {processed_count + failed_count} total grids processed")

            if service_restart_needed and failed_count > 0:
                logger.warning(f"S2 completed with {failed_count} failed grids.")


            if 'simple_hospitals_data_s2_ref' in locals() and simple_hospitals_data_s2_ref is not None:
                try: 
                    ray.experimental.delete_object_refs([simple_hospitals_data_s2_ref], True)
                except Exception: 
                    pass

            logger.info(f"S2 for {region_name_cn} completed: {processed_count} success, {failed_count} failed")

    stop_osm_ch_service()

    output_gpkg_filename = f"{region_id_for_filenames}_acc.gpkg"
    output_gpkg_path = os.path.join(config_params['RESULTS_DIR'], output_gpkg_filename)

    final_cols_to_save = ['grid_id', 'geometry', 'sum']
    if 'acc' not in grids_in_region_gdf.columns: grids_in_region_gdf['acc'] = 0.0
    grids_in_region_gdf['acc'] = grids_in_region_gdf['acc'].fillna(0.0)
    final_cols_to_save.append('acc')
    if 'nearest_L3_time' not in grids_in_region_gdf.columns: grids_in_region_gdf['nearest_L3_time'] = -1.0
    grids_in_region_gdf['nearest_L3_time'] = grids_in_region_gdf['nearest_L3_time'].fillna(-1.0)
    final_cols_to_save.append('nearest_L3_time')
    if not config_params['CALCULATE_LEVEL_3_ONLY']:
        if 'nearest_L2_time' not in grids_in_region_gdf.columns: grids_in_region_gdf['nearest_L2_time'] = -1.0
        grids_in_region_gdf['nearest_L2_time'] = grids_in_region_gdf['nearest_L2_time'].fillna(-1.0)
        final_cols_to_save.append('nearest_L2_time')
        if 'nearest_L1_time' not in grids_in_region_gdf.columns: grids_in_region_gdf['nearest_L1_time'] = -1.0
        grids_in_region_gdf['nearest_L1_time'] = grids_in_region_gdf['nearest_L1_time'].fillna(-1.0)
        final_cols_to_save.append('nearest_L1_time')

    actual_cols_to_save = [col for col in final_cols_to_save if col in grids_in_region_gdf.columns]
    if 'geometry' not in actual_cols_to_save and 'geometry' in grids_in_region_gdf.columns :
        if 'geometry' not in actual_cols_to_save: actual_cols_to_save.append('geometry')
    if not actual_cols_to_save or 'geometry' not in actual_cols_to_save:
        logger.error(f"Cannot save GPKG for {region_name_cn}: No columns or no geometry column to save. Columns available for saving: {actual_cols_to_save}")
        return

    grids_to_save = grids_in_region_gdf[actual_cols_to_save].copy()
    if grids_to_save.empty and 'geometry' not in grids_to_save.columns: # Handle case where grids_to_save might become empty AND lose geometry
        logger.warning(f"grids_to_save for {region_name_cn} is empty or missing geometry. Cannot save GPKG.")
        return

    if grids_to_save.crs is None: grids_to_save.crs = "EPSG:4326"

    try:
        grids_to_save.to_file(output_gpkg_path, driver="GPKG")
        logger.info(f"Successfully saved final results for {region_name_cn} to {output_gpkg_path}")
    except Exception as e:
        logger.error(f"Error saving final GPKG for {region_name_cn}: {e}", exc_info=True)



# --- Main Execution Block ---
# --- Main Execution Block ---
def run_pipeline():
    logger.info("Script started.")
    config = {
        'YEAR': YEAR, 'ENABLE_SAMPLING': ENABLE_SAMPLING, 'SAMPLE_FRACTION': SAMPLE_FRACTION,
        'CALCULATE_LEVEL_3_ONLY': CALCULATE_LEVEL_3_ONLY, 'RESULTS_DIR': RESULTS_DIR, 'TMP_DIR': TMP_DIR,
        'OSM_CH_WEB_BASE_URL': OSM_CH_WEB_BASE_URL, 'OSM_CH_WEB_STARTUP_TIME': OSM_CH_WEB_STARTUP_TIME,
        'OSM_CH_REQUEST_TIMEOUT': OSM_CH_REQUEST_TIMEOUT, 'OSM_CH_REQUEST_MAX_RETRIES_PER_CALL': OSM_CH_REQUEST_MAX_RETRIES_PER_CALL,
        'OSM_CH_SERVICE_RESTART_COOLDOWN': OSM_CH_SERVICE_RESTART_COOLDOWN
    }

    if not ray.is_initialized():
        os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
        os.environ["RAY_DEDUP_LOGS"] = "0"
        ray.init(
            ignore_reinit_error=True,
            log_to_driver=False,
            configure_logging=False,
            logging_level=logging.ERROR
        )
        warnings.filterwarnings("ignore", message=".*PYTHON worker processes have been started.*")
        warnings.filterwarnings("ignore", message=".*This could be a result of using a large number of actors.*")
        warnings.filterwarnings("ignore", category=UserWarning, module="ray")

    logger.info(f"Ray initialized: {ray.cluster_resources()}")

    try:
        china_boundaries_gdf = gpd.read_file(CHINA_BOUNDARIES_GEOJSON)
        logger.info(f"Loaded China boundaries: {len(china_boundaries_gdf)} regions.")
    except Exception as e:
        logger.error(f"Failed to load China boundaries GeoJSON {CHINA_BOUNDARIES_GEOJSON}: {e}", exc_info=True)
        if ray.is_initialized():
            ray.shutdown()
        return 1

    try:
        all_hospitals_df = pd.read_csv(HOSPITALS_DATA_CSV)
        required_cols = ['lng', 'lat', 'beds', 'grade']
        if not all(col in all_hospitals_df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in all_hospitals_df.columns]
            logger.error(f"Hospitals CSV {HOSPITALS_DATA_CSV} missing required columns: {missing_cols}. Available: {all_hospitals_df.columns.tolist()}")
            raise ValueError(f"Missing required columns in hospital data: {missing_cols}")
        all_hospitals_master_gdf = gpd.GeoDataFrame(
            all_hospitals_df,
            geometry=gpd.points_from_xy(all_hospitals_df["lng"], all_hospitals_df["lat"]),
            crs="EPSG:4326"
        )
        if 'hospital_id' not in all_hospitals_master_gdf.columns:
            all_hospitals_master_gdf['hospital_id'] = all_hospitals_master_gdf.index
        elif not all_hospitals_master_gdf['hospital_id'].is_unique:
            logger.warning("Hospital IDs from CSV are not unique. Using DataFrame index as hospital_id instead to ensure uniqueness.")
            all_hospitals_master_gdf['hospital_id'] = range(len(all_hospitals_master_gdf))
        logger.info(f"Loaded {len(all_hospitals_master_gdf)} hospitals.")
    except Exception as e:
        logger.error(f"Failed to load or process hospitals CSV {HOSPITALS_DATA_CSV}: {e}", exc_info=True)
        if ray.is_initialized():
            ray.shutdown()
        return 1

    if 'name' not in china_boundaries_gdf.columns:
        logger.error("name column not found in China boundaries GeoJSON. Cannot identify regions.")
        if ray.is_initialized():
            ray.shutdown()
        return 1

    try:
        for _, region_series in tqdm(china_boundaries_gdf.iterrows(), total=len(china_boundaries_gdf), desc="Processing Regions"):
            region_name = region_series["name"]
            current_region_gdf = gpd.GeoDataFrame([region_series], crs=china_boundaries_gdf.crs)
            logger.info(f"--- Processing region: {region_name} ---")

            final_region_output_path = os.path.join(config["RESULTS_DIR"], f"{region_name}_acc.gpkg")
            if os.path.exists(final_region_output_path):
                logger.info(f"Results for {region_name} already exist at {final_region_output_path}. Skipping.")
                continue

            try:
                process_region(region_name, current_region_gdf, all_hospitals_master_gdf, config)
            except Exception as e_proc_region:
                logger.error(f"Unhandled critical error processing region {region_name}: {e_proc_region}", exc_info=True)
                if osm_ch_service_process and osm_ch_service_process.poll() is None:
                    logger.warning(f"Service for {region_name} might still be running after unhandled error in process_region. Stopping.")
                    stop_osm_ch_service()
            finally:
                if osm_ch_service_process and osm_ch_service_process.poll() is None:
                    logger.warning(f"Service for {region_name} was still running after process_region completed/exited. Ensuring stop.")
                    stop_osm_ch_service()
    finally:
        if ray.is_initialized():
            logger.info("All regions processed. Shutting down Ray.")
            ray.shutdown()
        logger.info("Script finished.")

    return 0


def main():
    args = parse_args()
    apply_runtime_args(args)
    return run_pipeline()


if __name__ == "__main__":
    raise SystemExit(main())
