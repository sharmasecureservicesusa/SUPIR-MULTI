import os
import sys
import glob
import json
import time
import uuid
import shutil
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor

COMFYUI_DIR = os.getenv("COMFYUI_DIR", "/opt/ComfyUI")
if not os.path.exists(COMFYUI_DIR) and os.path.exists("/workspace/ComfyUI"):
    COMFYUI_DIR = "/workspace/ComfyUI"

SERVER_ADDRESS = "127.0.0.1:8188"
INPUT_DIR = "/mnt/s3bucket/input"
OUTPUT_DIR = "/mnt/s3bucket/output"
WORKFLOW_FILE = "/app/workflow_api.json"
COMFY_LOG_FILE = "/tmp/comfyui.log"
TEMP_LOCAL_DIR = "/tmp/batch_processing"

try:
    upscale_factor = float(os.getenv("UPSCALE_FACTOR", "3.0"))
except ValueError:
    upscale_factor = 3.0

def find_python_executable():
    candidates = [
        "/opt/environments/python/comfyui/bin/python",
        "/opt/environments/python/comfyui/bin/python3",
        "/opt/micromamba/envs/comfyui/bin/python",
        "/workspace/ComfyUI/venv/bin/python",
        "/opt/ComfyUI/venv/bin/python",
    ]
    for path in candidates:
        if os.path.exists(path):
            res = subprocess.run([path, "-c", "import torch"], capture_output=True)
            if res.returncode == 0:
                return path
    for pattern in ["/opt/**/bin/python*", "/workspace/**/bin/python*"]:
        for path in glob.glob(pattern, recursive=True):
            if os.path.isfile(path) and os.access(path, os.X_OK) and not path.endswith("-config"):
                res = subprocess.run([path, "-c", "import torch"], capture_output=True)
                if res.returncode == 0:
                    return path
    return sys.executable

def find_comfyui_dir():
    candidates = [os.getenv("COMFYUI_DIR", ""), "/workspace/ComfyUI", "/opt/ComfyUI", "/app/ComfyUI"]
    for path in candidates:
        if path and os.path.exists(os.path.join(path, "main.py")):
            return path
    return "/opt/ComfyUI"

def ensure_comfyui_running():
    try:
        urllib.request.urlopen(f"http://{SERVER_ADDRESS}/system_stats", timeout=2)
        print("✓ ComfyUI server is active.")
        return None
    except Exception:
        print("Launching local ComfyUI instance with High-VRAM + SDPA optimizations...")
        python_bin = find_python_executable()
        comfy_dir = find_comfyui_dir()
        main_py = os.path.join(comfy_dir, "main.py")
        
        log_handle = open(COMFY_LOG_FILE, "w")
        proc = subprocess.Popen(
            [
                python_bin, main_py,
                "--listen", "127.0.0.1",
                "--port", "8188",
                "--highvram",
                "--fp16-vae",
                "--use-pytorch-cross-attention",
                "--disable-auto-launch"
            ],
            stdout=log_handle,
            stderr=subprocess.STDOUT
        )

        for _ in range(60):
            if proc.poll() is not None:
                print("❌ ComfyUI process exited prematurely.")
                os.system(f"cat {COMFY_LOG_FILE}")
                sys.exit(1)
            try:
                urllib.request.urlopen(f"http://{SERVER_ADDRESS}/system_stats", timeout=2)
                print("✓ ComfyUI server initialized successfully!")
                return proc
            except Exception:
                time.sleep(1)
        sys.exit(1)

def queue_prompt(prompt_workflow):
    client_id = str(uuid.uuid4())
    payload = json.dumps({"prompt": prompt_workflow, "client_id": client_id}).encode("utf-8")
    req = urllib.request.Request(
        f"http://{SERVER_ADDRESS}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as response:
        res_data = json.loads(response.read().decode("utf-8"))
        return res_data["prompt_id"]

def wait_for_completion(prompt_id):
    while True:
        try:
            with urllib.request.urlopen(f"http://{SERVER_ADDRESS}/history/{prompt_id}") as resp:
                history = json.loads(resp.read().decode("utf-8"))
                if prompt_id in history:
                    return history[prompt_id]
        except Exception:
            pass
        time.sleep(0.1)

def process_single_image(img_name, base_workflow, load_image_node, comfy_input_dir):
    src_input_path = os.path.join(INPUT_DIR, img_name)
    target_output_path = os.path.join(OUTPUT_DIR, f"upscaled_{img_name}")
    comfy_temp_input = os.path.join(comfy_input_dir, img_name)

    try:
        shutil.copy(src_input_path, comfy_temp_input)

        current_workflow = json.loads(json.dumps(base_workflow))
        if load_image_node:
            current_workflow[load_image_node]["inputs"]["image"] = img_name

        prompt_id = queue_prompt(current_workflow)
        history = wait_for_completion(prompt_id)
        outputs = history.get("outputs", {})
        generated_full_path = None

        for node_id, output_data in outputs.items():
            if "images" in output_data and len(output_data["images"]) > 0:
                img_info = output_data["images"][0]
                generated_full_path = os.path.join(COMFYUI_DIR, "output", img_info.get("subfolder", ""), img_info["filename"])
                break

        if generated_full_path and os.path.exists(generated_full_path):
            shutil.move(generated_full_path, target_output_path)
            if os.path.exists(comfy_temp_input):
                os.remove(comfy_temp_input)
            if os.path.exists(src_input_path):
                os.remove(src_input_path)
            print(f"✓ Successfully processed: {img_name}")
        else:
            print(f"⚠️ Output missing for {img_name}")

    except Exception as e:
        print(f"❌ Error processing {img_name}: {str(e)}")

def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_LOCAL_DIR, exist_ok=True)

    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    if not image_files:
        print(f"No images found in {INPUT_DIR}. Exiting.")
        sys.exit(0)

    print(f"Found {len(image_files)} images to process.")
    server_proc = ensure_comfyui_running()

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

    # Parallelize file preparation and queue submissions
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(process_single_image, img_name, base_workflow, load_image_node, comfy_input_dir)
            for img_name in image_files
        ]
        for future in futures:
            future.result()

    elapsed = time.time() - start_time
    print(f"\n=== Completed {len(image_files)} images in {elapsed:.2f}s ({len(image_files)/elapsed:.2f} img/sec) ===")

    if server_proc:
        server_proc.terminate()

if __name__ == "__main__":
    main()