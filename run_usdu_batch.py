import os
import sys
import time
import glob
import json
import shutil
import subprocess
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

COMFYUI_DIR = "/opt/ComfyUI"
PYTHON_BIN = "/opt/environments/python/comfyui/bin/python3"
if not os.path.exists(PYTHON_BIN):
    PYTHON_BIN = sys.executable

INPUT_S3_DIR = "/mnt/s3bucket/input"
OUTPUT_S3_DIR = "/mnt/s3bucket/output"
RAM_INPUT_DIR = "/dev/shm/batch_input"
RAM_OUTPUT_DIR = "/dev/shm/batch_output"

os.makedirs(RAM_INPUT_DIR, exist_ok=True)
os.makedirs(RAM_OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_S3_DIR, exist_ok=True)

def start_comfyui_worker(gpu_id, port):
    cmd = [
        PYTHON_BIN, os.path.join(COMFYUI_DIR, "main.py"),
        "--port", str(port),
        "--dont-print-server"
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc

def wait_for_server(port, timeout=120):
    start = time.time()
    url = f"http://127.0.0.1:{port}/history"
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False

def build_supir_workflow(input_filename, output_prefix):
    return {
        "3": {
            "inputs": {
                "image": input_filename,
                "upload": "image"
            },
            "class_type": "LoadImage"
        },
        "4": {
            "inputs": {
                "ckpt_name": "sd_xl_base_1.0.safetensors"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "5": {
            "inputs": {
                "supir_model": "SUPIR-v0F.ckpt",
                "fp8_unet": False
            },
            "class_type": "SUPIR_model_loader"
        },
        "6": {
            "inputs": {
                "use_tiled": True,
                "tile_size": 1024,
                "tile_stride": 512,
                "steps": 20,
                "cfg": 4.0,
                "min_cfg": 1.0,
                "sampler_name": "euler_ancestral",
                "supir_model": ["5", 0],
                "model": ["4", 0],
                "clip": ["4", 1],
                "vae": ["4", 2],
                "image": ["3", 0]
            },
            "class_type": "SUPIR_sample"
        },
        "7": {
            "inputs": {
                "filename_prefix": output_prefix,
                "images": ["6", 0]
            },
            "class_type": "SaveImage"
        }
    }

def process_single_image(img_path, worker_port):
    filename = os.path.basename(img_path)
    output_prefix = f"upscaled_{os.path.splitext(filename)[0]}"
    
    # 1. Copy image to ComfyUI input folder
    comfy_input_path = os.path.join(COMFYUI_DIR, "input", filename)
    shutil.copy2(img_path, comfy_input_path)

    # 2. Construct workflow JSON
    workflow = build_supir_workflow(filename, output_prefix)
    payload = json.dumps({"prompt": workflow}).encode('utf-8')
    
    req = urllib.request.Request(
        f"http://127.0.0.1:{worker_port}/prompt",
        data=payload,
        headers={'Content-Type': 'application/json'}
    )

    try:
        with urllib.request.urlopen(req) as resp:
            res_data = json.loads(resp.read().decode('utf-8'))
            prompt_id = res_data.get('prompt_id')
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        print(f"❌ ComfyUI Validation Error on {filename} (HTTP {e.code}):\n{error_body}")
        raise e

    # 3. Poll execution status
    history_url = f"http://127.0.0.1:{worker_port}/history/{prompt_id}"
    while True:
        try:
            with urllib.request.urlopen(history_url) as resp:
                history = json.loads(resp.read().decode('utf-8'))
                if prompt_id in history:
                    status = history[prompt_id].get('status', {})
                    if status.get('completed', False) or status.get('status_str') == 'success':
                        break
                    if status.get('status_str') == 'error':
                        messages = status.get('messages', [])
                        raise RuntimeError(f"Execution Error in ComfyUI: {messages}")
        except Exception as ex:
            if "Execution Error" in str(ex):
                raise ex
        time.sleep(1)

    # 4. Copy produced output image back to S3 destination
    comfy_output_pattern = os.path.join(COMFYUI_DIR, "output", f"{output_prefix}*")
    produced_files = glob.glob(comfy_output_pattern)
    if produced_files:
        latest_file = max(produced_files, key=os.path.getctime)
        final_s3_dest = os.path.join(OUTPUT_S3_DIR, os.path.basename(latest_file))
        shutil.copy2(latest_file, final_s3_dest)
        os.remove(latest_file)
    
    if os.path.exists(comfy_input_path):
        os.remove(comfy_input_path)

def main():
    print("=== Detecting Available Hardware ===" )
    try:
        nvidia_smi = subprocess.check_output(["nvidia-smi", "-L"]).decode('utf-8')
        gpu_count = len([line for line in nvidia_smi.strip().split('\n') if line])
    except Exception:
        gpu_count = 1

    print(f"Detected GPUs: {gpu_count}")
    
    ports = []
    procs = []
    base_port = 8188

    # Spawn 1 worker instance per GPU
    for gpu_id in range(gpu_count):
        port = base_port + gpu_id
        print(f"Launching ComfyUI Worker on GPU {gpu_id} (Port {port})...")
        proc = start_comfyui_worker(gpu_id, port)
        procs.append(proc)
        ports.append(port)

    for port in ports:
        if not wait_for_server(port):
            print(f"❌ Failed to initialize worker on port {port}")
            for p in procs:
                p.terminate()
            sys.exit(1)
        print(f"✓ Active Worker -> Port {port}")

    # Gather images from input path
    raw_images = glob.glob(os.path.join(INPUT_S3_DIR, "*.[jJ][pP][gG]")) + \
                 glob.glob(os.path.join(INPUT_S3_DIR, "*.[pP][nN][gG]")) + \
                 glob.glob(os.path.join(INPUT_S3_DIR, "*.[wW][eE][bB][pP]"))

    if not raw_images:
        print(f"No input images found in {INPUT_S3_DIR}. Batch job complete.")
        for p in procs:
            p.terminate()
        return

    print(f"Syncing {len(raw_images)} images to RAM Disk...")
    ram_images = []
    for img in raw_images:
        dest = os.path.join(RAM_INPUT_DIR, os.path.basename(img))
        shutil.copy2(img, dest)
        ram_images.append(dest)

    start_time = time.time()
    processed_count = 0

    with ThreadPoolExecutor(max_workers=len(ports)) as executor:
        futures = {}
        for idx, img_path in enumerate(ram_images):
            assigned_port = ports[idx % len(ports)]
            fut = executor.submit(process_single_image, img_path, assigned_port)
            futures[fut] = img_path

        for fut in as_completed(futures):
            img_path = futures[fut]
            try:
                fut.result()
                processed_count += 1
                print(f"✓ Completed: {os.path.basename(img_path)}")
            except Exception as e:
                print(f"❌ Error processing {os.path.basename(img_path)}: {e}")

    elapsed = time.time() - start_time
    rate = processed_count / elapsed if elapsed > 0 else 0
    print(f"=== Completed {processed_count} images in {elapsed:.2f}s ({rate:.2f} img/sec) ===")

    for p in procs:
        p.terminate()

if __name__ == "__main__":
    main()