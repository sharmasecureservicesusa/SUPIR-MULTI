import os
import sys
import glob
import json
import time
import uuid
import shutil
import asyncio
import itertools
import subprocess
import urllib.request
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

WORKFLOW_FILE = "/app/workflow_api.json"

# Launch 4 Workers: 2 per GPU to fully utilize dual L40S Tensor Cores
WORKERS = [
    {"gpu_id": 0, "port": 8188},
    {"gpu_id": 0, "port": 8189},
    {"gpu_id": 1, "port": 8190},
    {"gpu_id": 1, "port": 8191},
]

worker_processes = []
worker_ports = [w["port"] for w in WORKERS]
port_cycle = itertools.cycle(worker_ports)

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

def find_comfyui_dir():
    candidates = [os.getenv("COMFYUI_DIR", ""), "/workspace/ComfyUI", "/opt/ComfyUI", "/app/ComfyUI"]
    for path in candidates:
        if path and os.path.exists(os.path.join(path, "main.py")):
            return path
    return "/opt/ComfyUI"

def start_comfy_worker(gpu_id, port):
    python_bin = find_python_executable()
    comfy_dir = find_comfyui_dir()
    main_py = os.path.join(comfy_dir, "main.py")
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
            print(f"❌ ComfyUI worker failed on GPU {gpu_id} (Port {port})")
            sys.exit(1)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=2)
            print(f"✓ ComfyUI active on GPU {gpu_id} (Port {port})")
            return proc
        except Exception:
            time.sleep(1)
    sys.exit(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker_processes
    print("🚀 Initializing Multi-GPU ComfyUI Worker Pool...")
    for worker in WORKERS:
        proc = start_comfy_worker(worker["gpu_id"], worker["port"])
        worker_processes.append(proc)
    yield
    print("Terminating ComfyUI worker processes...")
    for proc in worker_processes:
        proc.terminate()

app = FastAPI(title="ComfyUI Dual-L40S Max Speed Endpoint", lifespan=lifespan)

def queue_prompt(port, prompt_workflow):
    client_id = str(uuid.uuid4())
    payload = json.dumps({"prompt": prompt_workflow, "client_id": client_id}).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{port}/prompt", data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as response:
        res_data = json.loads(response.read().decode("utf-8"))
        return res_data["prompt_id"]

async def wait_for_completion(port, prompt_id):
    while True:
        try:
            req_url = f"http://127.0.0.1:{port}/history/{prompt_id}"
            with urllib.request.urlopen(req_url) as resp:
                history = json.loads(resp.read().decode("utf-8"))
                if prompt_id in history:
                    return history[prompt_id]
        except Exception:
            pass
        await asyncio.sleep(0.05)

def cleanup_file(filepath: str):
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception:
            pass

@app.get("/health")
def health():
    return {"status": "ready", "active_workers": len(worker_processes)}

@app.post("/v1/upscale")
async def upscale(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    comfy_dir = find_comfyui_dir()
    comfy_input_dir = os.path.join(comfy_dir, "input")
    os.makedirs(comfy_input_dir, exist_ok=True)

    filename = f"{uuid.uuid4()}_{file.filename}"
    temp_input_path = os.path.join(comfy_input_dir, filename)

    with open(temp_input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Select next worker port via round-robin
    assigned_port = next(port_cycle)

    try:
        with open(WORKFLOW_FILE, "r") as f:
            workflow = json.load(f)

        for node_id, node in workflow.items():
            if node.get("class_type") == "LoadImage":
                workflow[node_id]["inputs"]["image"] = filename

        prompt_id = queue_prompt(assigned_port, workflow)
        history = await wait_for_completion(assigned_port, prompt_id)

        outputs = history.get("outputs", {})
        generated_path = None

        for node_id, output_data in outputs.items():
            if "images" in output_data and len(output_data["images"]) > 0:
                img_info = output_data["images"][0]
                generated_path = os.path.join(comfy_dir, "output", img_info.get("subfolder", ""), img_info["filename"])
                break

        if generated_path and os.path.exists(generated_path):
            # Schedule file cleanup after response is sent
            background_tasks.add_task(cleanup_file, generated_path)
            background_tasks.add_task(cleanup_file, temp_input_path)
            
            return FileResponse(generated_path, media_type="image/png", filename=f"upscaled_{file.filename}")
        else:
            raise HTTPException(status_code=500, detail="Upscaling failed to generate output image.")

    except Exception as e:
        cleanup_file(temp_input_path)
        raise HTTPException(status_code=500, detail=str(e))