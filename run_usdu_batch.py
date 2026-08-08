import os
import sys
import glob
import json
import time
import uuid
import signal
import shutil
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor

COMFYUI_DIR = os.getenv("COMFYUI_DIR", "/opt/ComfyUI")
if not os.path.exists(COMFYUI_DIR) and os.path.exists("/workspace/ComfyUI"):
    COMFYUI_DIR = "/workspace/ComfyUI"

S3_INPUT_DIR = "/mnt/s3bucket/input"
S3_OUTPUT_DIR = "/mnt/s3bucket/output"
RAM_INPUT_DIR = "/dev/shm/batch_input"
RAM_OUTPUT_DIR = "/dev/shm/batch_output"
WORKFLOW_FILE = "/app/workflow_api.json"

# Define 4 Workers: 2 processes on GPU 0, 2 processes on GPU 1
WORKERS = [
    {"gpu_id": 0, "port": 8188},
    {"gpu_id": 0, "port": 8189},
    {"gpu_id": 1, "port": 8190},
    {"gpu_id": 1, "port": 8191},
]

shutdown_requested = False

def handle_sigterm(signum, frame):
    global shutdown_requested
    print("\n⚠️ Preemptible signal (SIGTERM) received! Flushing outputs and shutting down cleanly...")
    shutdown_requested = True

signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

def find_python_executable():
    candidates = [
        "/opt/environments/python/comfyui/bin/python3",
        "/opt/environments/python/comfyui/bin/python",
        "/workspace/ComfyUI/venv/bin/python",
        "/opt/ComfyUI/venv/bin/python",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return sys.executable

def start_worker(gpu_id, port):
    python_bin = find_python_executable()
    main_py = os.path.join(COMFYUI_DIR, "main.py")
    log_file = f"/tmp/comfyui_gpu{gpu_id}_port{port}.log"
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_handle = open(log_file, "w")
    proc = subprocess.Popen(
        [
            python_bin, main_py,
            "--listen", "127.0.0.1",
            "--port", str(port),
            "--highvram",
            "--fp16-vae",
            "--use-pytorch-cross-attention",
            "--disable-auto-launch"
        ],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env
    )

    for _ in range(60):
        if proc.poll() is not None:
            print(f"❌ ComfyUI worker on GPU {gpu_id}:{port} failed.")
            sys.exit(1)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=2)
            print(f"✓ Active Worker -> GPU {gpu_id} | Port {port}")
            return proc
        except Exception:
            time.sleep(1)
    sys.exit(1)

def queue_prompt(port, prompt_workflow):
    client_id = str(uuid.uuid4())
    payload = json.dumps({"prompt": prompt_workflow, "client_id": client_id}).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as response:
        res_data = json.loads(response.read().decode("utf-8"))
        return res_data["prompt_id"]

def wait_for_completion(port, prompt_id):
    while True:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/history/{prompt_id}") as resp:
                history = json.loads(resp.read().decode("utf-8"))
                if prompt_id in history:
                    return history[prompt_id]
        except Exception:
            pass
        time.sleep(0.05)

def process_single_image(img_name, base_workflow, load_image_node, comfy_input_dir, task_index):
    if shutdown_requested:
        return

    worker = WORKERS[task_index % len(WORKERS)]
    port = worker["port"]
    
    ram_src_path = os.path.join(RAM_INPUT_DIR, img_name)
    ram_dest_path = os.path.join(RAM_OUTPUT_DIR, f"upscaled_{img_name}")
    s3_dest_path = os.path.join(S3_OUTPUT_DIR, f"upscaled_{img_name}")
    s3_src_path = os.path.join(S3_INPUT_DIR, img_name)
    comfy_temp_input = os.path.join(comfy_input_dir, img_name)

    try:
        shutil.copy(ram_src_path, comfy_temp_input)

        current_workflow = json.loads(json.dumps(base_workflow))
        if load_image_node:
            current_workflow[load_image_node]["inputs"]["image"] = img_name

        prompt_id = queue_prompt(port, current_workflow)
        history = wait_for_completion(port, prompt_id)
        outputs = history.get("outputs", {})
        generated_full_path = None

        for node_id, output_data in outputs.items():
            if "images" in output_data and len(output_data["images"]) > 0:
                img_info = output_data["images"][0]
                generated_full_path = os.path.join(COMFYUI_DIR, "output", img_info.get("subfolder", ""), img_info["filename"])
                break

        if generated_full_path and os.path.exists(generated_full_path):
            # Move result to RAM disk first
            shutil.move(generated_full_path, ram_dest_path)
            # Sync RAM disk output to S3 mount
            shutil.copy(ram_dest_path, s3_dest_path)
            
            # Clean up local temporary files
            if os.path.exists(comfy_temp_input):
                os.remove(comfy_temp_input)
            if os.path.exists(s3_src_path):
                os.remove(s3_src_path)
            print(f"✓ Processed {img_name} on GPU {worker['gpu_id']} (Port {port})")

    except Exception as e:
        print(f"❌ Error processing {img_name}: {str(e)}")

def main():
    os.makedirs(S3_INPUT_DIR, exist_ok=True)
    os.makedirs(S3_OUTPUT_DIR, exist_ok=True)
    os.makedirs(RAM_INPUT_DIR, exist_ok=True)
    os.makedirs(RAM_OUTPUT_DIR, exist_ok=True)

    s3_files = [f for f in os.listdir(S3_INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    if not s3_files:
        print("No input images found. Exiting.")
        sys.exit(0)

    print(f"Syncing {len(s3_files)} images to 384GB System RAM disk (`/dev/shm`).")
    for f in s3_files:
        shutil.copy(os.path.join(S3_INPUT_DIR, f), os.path.join(RAM_INPUT_DIR, f))

    print(f"Initializing 4x ComfyUI workers across 2x L40S GPUs...")
    processes = [start_worker(w["gpu_id"], w["port"]) for w in WORKERS]

    with open(WORKFLOW_FILE, "r") as f:
        base_workflow = json.load(f)

    load_image_node = None
    for node_id, node_data in base_workflow.items():
        if node_data.get("class_type") == "LoadImage":
            load_image_node = node_id
            break

    comfy_input_dir = os.path.join(COMFYUI_DIR, "input")
    os.makedirs(comfy_input_dir, exist_ok=True)

    start_time = time.time()

    # Utilize 16 CPU worker threads to maximize 64 vCPU scheduling
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(process_single_image, img_name, base_workflow, load_image_node, comfy_input_dir, idx)
            for idx, img_name in enumerate(s3_files)
        ]
        for future in futures:
            future.result()

    elapsed = time.time() - start_time
    rate = len(s3_files) / elapsed if elapsed > 0 else 0
    print(f"\n=== Completed {len(s3_files)} images in {elapsed:.2f}s ({rate:.2f} img/sec | {rate*60:.2f} img/min) ===")

    for proc in processes:
        proc.terminate()

    # Cleanup RAM disk
    shutil.rmtree(RAM_INPUT_DIR, ignore_errors=True)
    shutil.rmtree(RAM_OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()