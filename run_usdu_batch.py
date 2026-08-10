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
    log_file_path = f"/tmp/comfyui_gpu_{gpu_id}_port_{port}.log"
    log_file = open(log_file_path, "w")
    
    cmd = [
        PYTHON_BIN, os.path.join(COMFYUI_DIR, "main.py"),
        "--port", str(port),
        "--listen", "127.0.0.1",
        "--dont-print-server"
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_file_path, log_file

def wait_for_server(proc, log_file_path, port, timeout=120):
    start = time.time()
    url = f"http://127.0.0.1:{port}/system_stats"
    
    while time.time() - start < timeout:
        if proc.poll() is not None:
            print(f"❌ ComfyUI process on port {port} crashed with exit code {proc.returncode}!")
            if os.path.exists(log_file_path):
                with open(log_file_path, "r") as f:
                    print(f"=== Startup Log (Port {port}) ===\n{f.read()}")
            return False

        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(2)

    print(f"❌ Timeout waiting for ComfyUI worker on port {port}.")
    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as f:
            print(f"=== Startup Log (Port {port}) ===\n{f.read()}")
    return False

def build_supir_workflow(input_filename, output_prefix):
    return {
        "1": {
            "inputs": {
                "ckpt_name": "sd_xl_base_1.0.safetensors"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "2": {
            "inputs": {
                "supir_model": "SUPIR-v0F.ckpt",
                "sdxl_model": "sd_xl_base_1.0.safetensors",
                "fp8_unet": True
            },
            "class_type": "SUPIR_model_loader"
        },
        "3": {
            "inputs": {
                "image": input_filename
            },
            "class_type": "LoadImage"
        },
        "4": {
            "inputs": {
                "text": "high quality, detailed photo, sharp textures, clean skin, natural lighting",
                "clip": ["1", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "5": {
            "inputs": {
                "text": "bad quality, blurry, pixelated, noise, artifacts, distorted faces",
                "clip": ["1", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "10": {
            "inputs": {
                "upscale_method": "bicubic",
                "scale_by": 3.0,
                "image": ["3", 0]
            },
            "class_type": "ImageScaleBy"
        },
        "6": {
            "inputs": {
                "SUPIR_VAE": ["2", 1],
                "image": ["10", 0],
                "use_tiled_vae": True,
                "encoder_tile_size": 1024,
                "decoder_tile_size": 1024
            },
            "class_type": "SUPIR_first_stage"
        },
        "7": {
            "inputs": {
                "seed": 123456789,
                "steps": 12,
                "cfg_scale": 4.0,
                "min_cfg_scale": 1.0,
                "s_churn": 0.0,
                "s_noise": 1.003,
                "s_cfg": 1.0,
                "restoration_scale": 1.0,
                "SUPIR_model": ["2", 0],
                "latents": ["6", 2],
                "positive": ["4", 0],
                "negative": ["5", 0]
            },
            "class_type": "SUPIR_sample"
        },
        "8": {
            "inputs": {
                "SUPIR_VAE": ["2", 1],
                "latents": ["7", 0],
                "use_tiled_vae": True,
                "decoder_tile_size": 1024
            },
            "class_type": "SUPIR_decode"
        },
        "9": {
            "inputs": {
                "filename_prefix": output_prefix,
                "images": ["8", 0]
            },
            "class_type": "SaveImage"
        }
    }

def process_single_image(img_path, worker_port):
    filename = os.path.basename(img_path)
    output_prefix = f"supir_restored_{os.path.splitext(filename)[0]}"
    
    comfy_input_path = os.path.join(COMFYUI_DIR, "input", filename)
    shutil.copy2(img_path, comfy_input_path)

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
    print("=== Detecting Available Hardware ===")
    try:
        nvidia_smi = subprocess.check_output(["nvidia-smi", "-L"]).decode('utf-8')
        gpu_count = len([line for line in nvidia_smi.strip().split('\n') if line])
    except Exception:
        gpu_count = 1

    print(f"Detected GPUs: {gpu_count}")
    
    ports = []
    procs = []
    log_files = []
    base_port = 8188

    for gpu_id in range(gpu_count):
        port = base_port + gpu_id
        print(f"Launching ComfyUI Worker on GPU {gpu_id} (Port {port})...")
        proc, log_path, log_handle = start_comfyui_worker(gpu_id, port)
        procs.append(proc)
        ports.append(port)
        log_files.append((log_path, log_handle))

    for idx, port in enumerate(ports):
        proc = procs[idx]
        log_path = log_files[idx][0]
        if not wait_for_server(proc, log_path, port):
            print(f"❌ Aborting batch execution due to worker startup failure on port {port}.")
            for p in procs:
                p.terminate()
            sys.exit(1)
        print(f"✓ Active Worker -> GPU {idx} | Port {port}")

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
    for _, handle in log_files:
        handle.close()

if __name__ == "__main__":
    main()