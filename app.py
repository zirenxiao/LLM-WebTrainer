import sys
import os
import json
import re
import time
import uuid
import shutil
from flask import (
    Flask, request, render_template,
    jsonify, Response, redirect, url_for, flash
)
from werkzeug.utils import secure_filename
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, Dataset
import torch
from multiprocessing import Process

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "supersecretkey")

# ensure folders exist
os.makedirs("datasets", exist_ok=True)
os.makedirs("logs", exist_ok=True)
BASE_OUTPUT_DIR = "result"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

available_devices = [f"{i}" for i in range(torch.cuda.device_count())]
default_config = {
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "text_mapping": "messages",
    "batch_size": 1,
    "epochs": 10,
    "learning_rate": 5e-5,
    "DATA_PATH": "",
    "use_lora": True,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "padding_strategy": "max_length",
    "padding_max_length": 512,
    "eval_dataset_percentage": 0.2,
    "eval_strategy": "epoch",
    "eval_steps": 100,
    "weight_decay": 0.01,
    "logging_steps": 10,
    "logging_strategy": "epoch",
    "save_strategy": "epoch",
    "save_steps": 100,
    "save_total_limit": 5,
    "cuda_devices": ",".join(available_devices),
}
config = default_config.copy()

datasets_list = sorted(os.listdir("datasets"))
if datasets_list:
    config["DATA_PATH"] = os.path.join("datasets", datasets_list[0])

LOG_FILE = os.path.join("logs", "flask_logs.txt")
STATUS_FILE = os.path.join("logs", "status.json")
open(LOG_FILE, "a").close()
open(STATUS_FILE, "a").close()


def write_status(data):
    with open(STATUS_FILE, "w") as f:
        if isinstance(data, dict):
            json.dump(data, f)
        else:
            json.dump({"status": data}, f)


original_stdout = sys.stdout
original_stderr = sys.stderr


def write_log(msg: str):
    original_stdout.write(msg)
    original_stdout.flush()
    with open(LOG_FILE, "a") as f:
        for line in re.split(r"\r\n|\r|\n", msg):
            if line.strip(): f.write(line + "\n")


class ProgressCallback(TrainerCallback):
    def __init__(self, status_file, start_time):
        self.status_file = status_file
        self.start_time = start_time

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        current = state.global_step
        total = state.max_steps or 1
        pct = current / total * 100
        elapsed = time.time() - self.start_time
        remaining = (elapsed * (total - current) / current) if current else 0.0
        speed = (current / elapsed) if elapsed else 0.0

        def fmt(t):
            t = int(t);
            h, rem = divmod(t, 3600);
            m, s = divmod(rem, 60)
            return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

        progress = {"percentage": pct, "current_step": current, "total_steps": total,
                    "elapsed": fmt(elapsed), "remaining": fmt(remaining), "speed": speed}
        write_status({"status": "🚀 Training in progress...", "progress": progress})


class EvalCallback(TrainerCallback):
    def __init__(self, status_file, eval_steps):
        self.status_file = status_file
        self.eval_steps = eval_steps
        self.count = 0

    def on_prediction_step(self, args, state, control, **kwargs):
        self.count += 1
        pct = self.count / self.eval_steps * 100
        write_status({"status": "🔍 (In Train) Evaluating...",
                      "progress": {"percentage": pct, "current_step": self.count, "total_steps": self.eval_steps}})


def train_model(configs: dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = configs["cuda_devices"]

    class Logger:
        def write(self, m): write_log(m)

        def flush(self): pass

    sys.stdout = Logger();
    sys.stderr = Logger()

    write_status("⏳ Queued")
    try:
        write_log("=== Training started ===\n")
        write_log(f"Loading dataset from {configs['DATA_PATH']}...\n")
        ds = load_dataset("json", data_files=configs["DATA_PATH"])
        write_log("Dataset loaded.\n")

        write_log("Initializing tokenizer & model...\n")
        kwargs = {}
        if "gemma" in configs["model_name"]: kwargs["attn_implementation"] = "eager"
        if len(configs["cuda_devices"].split(",")) > 1: kwargs["device_map"] = "auto"
        tokenizer = AutoTokenizer.from_pretrained(configs["model_name"], **kwargs)
        model = AutoModelForCausalLM.from_pretrained(configs["model_name"], **kwargs)
        write_log("Model ready.\n")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            write_log("⚙️ pad_token was not set; using eos_token as pad_token.\n")

        if configs["use_lora"]:
            peft_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=configs["lora_r"],
                                  lora_alpha=configs["lora_alpha"], lora_dropout=configs["lora_dropout"],
                                  target_modules=["q_proj", "v_proj"])
            model = get_peft_model(model, peft_cfg)
            write_log("LoRA applied.\n")

        # Tokenizing with progress updates
        write_log("Tokenizing...\n")
        data = ds["train"]
        total = len(data)
        tokenized_list = []
        start_t = time.time()
        for idx, ex in enumerate(data):
            inputs = tokenizer(json.dumps(ex[configs["text_mapping"]]),
                               padding=configs["padding_strategy"],
                               truncation=True, max_length=configs["padding_max_length"])
            inputs["labels"] = inputs["input_ids"].copy()
            tokenized_list.append(inputs)
            pct = (idx + 1) / total * 100
            elapsed = time.time() - start_t
            rem = elapsed * (total - (idx + 1)) / (idx + 1)
            speed = (idx + 1) / elapsed
            write_status({"status": "🔄 Tokenizing dataset...",
                          "progress": {"percentage": pct, "current_step": idx + 1, "total_steps": total,
                                       "elapsed": f"{int(elapsed)}s", "remaining": f"{int(rem)}s", "speed": speed}})
        tokenized = Dataset.from_list(tokenized_list)
        write_log("Tokenization done.\n")

        splits = tokenized.train_test_split(test_size=configs["eval_dataset_percentage"])

        write_log("Initializing TrainingArguments...\n")
        args = TrainingArguments(
            output_dir=configs["output_dir"], eval_strategy=configs["eval_strategy"],
            learning_rate=configs["learning_rate"], per_device_train_batch_size=configs["batch_size"],
            per_device_eval_batch_size=configs["batch_size"], num_train_epochs=configs["epochs"],
            weight_decay=configs["weight_decay"], logging_dir=configs["logging_dir"],
            logging_steps=configs["logging_steps"], logging_strategy=configs["logging_strategy"],
            save_strategy=configs["save_strategy"], save_total_limit=configs["save_total_limit"], disable_tqdm=True
        )
        if configs["eval_strategy"] == "steps": args.eval_steps = configs["eval_steps"]
        if configs["save_strategy"] == "steps": args.save_steps = configs["save_steps"]

        start_time = time.time()
        trainer = Trainer(
            model=model, args=args,
            train_dataset=splits["train"], eval_dataset=splits["test"],
            callbacks=[ProgressCallback(STATUS_FILE, start_time), EvalCallback(STATUS_FILE, len(splits["test"]))]
        )

        write_log("Entering training loop...\n")
        trainer.train()
        write_log("Training loop ended.\n")

        write_log("Starting evaluation...\n")
        results = trainer.evaluate()
        write_log(f"=== Training completed: {results} ===\n")
        write_status("✅ Completed")
    except Exception as e:
        write_log(f"❌ ERROR: {e}\n")
        write_status("❌ Failed")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

@app.route("/", methods=["GET"])
def index():
    datasets = sorted(os.listdir("datasets"))
    runs = []
    for run in sorted(os.listdir(BASE_OUTPUT_DIR)):
        run_path = os.path.join(BASE_OUTPUT_DIR, run)
        if os.path.isdir(run_path):
            checkpoints = sorted(
                d for d in os.listdir(run_path)
                if os.path.isdir(os.path.join(run_path, d))
            )
            runs.append({"name": run, "checkpoints": checkpoints})
    return render_template(
        "index.html",
        config=config,
        devices=available_devices,
        datasets=datasets,
        runs=runs
    )

@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():
    file = request.files.get("dataset_file")
    if file and file.filename:
        fname = secure_filename(file.filename)
        file.save(os.path.join("datasets", fname))
        flash(f"✅ Dataset uploaded: {fname}")
    else:
        flash("❌ No file selected for upload.")
    return redirect(url_for("index"))

@app.route("/delete_dataset", methods=["POST"])
def delete_dataset():
    name = request.form.get("dataset_name")
    path = os.path.join("datasets", name)
    if name and os.path.exists(path):
        os.remove(path)
        flash(f"🗑️ Dataset deleted: {name}")
    else:
        flash(f"❌ Could not find dataset: {name}")
    return redirect(url_for("index"))

@app.route("/train", methods=["POST"])
def train():
    for k in config:
        if k == "DATA_PATH":
            continue
        if k in request.form and request.form[k]:
            v = request.form[k]
            try:
                config[k] = int(v)
            except ValueError:
                try:
                    config[k] = float(v)
                except ValueError:
                    config[k] = v
    selected = request.form.get("dataset")
    if selected:
        config["DATA_PATH"] = os.path.join("datasets", selected)

    run_id = time.strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:6]
    run_output_dir = os.path.join(BASE_OUTPUT_DIR, run_id)
    os.makedirs(run_output_dir, exist_ok=True)
    config["output_dir"]  = run_output_dir
    config["logging_dir"] = os.path.join(run_output_dir, "logs")

    global training_process
    if 'training_process' in globals() and training_process.is_alive():
        return jsonify(status="❌ Already running"), 400

    open(LOG_FILE, "w").close()
    write_status("⏳ Queued")

    training_process = Process(target=train_model, args=(config,))
    training_process.start()
    return jsonify(status="🚀 Training started")

@app.route("/stop", methods=["POST"])
def stop():
    global training_process
    if 'training_process' in globals() and training_process.is_alive():
        training_process.terminate()
        training_process.join()
        write_status("⏹️ Stopped")
        return jsonify(status="⏹️ Force killed"), 200
    return jsonify(status="❌ No active training"), 400

@app.route("/clear_logs", methods=["POST"])
def clear_logs():
    open(LOG_FILE, "w").close()
    return jsonify(status="🧹 Logs cleared")

@app.route("/status")
def status():
    default = {"status": "🤖 Idle"}
    try:
        data = json.load(open(STATUS_FILE))
    except:
        data = default
    return jsonify(data)

@app.route("/logs")
def logs():
    def generate():
        open(LOG_FILE, "a").close()
        with open(LOG_FILE, "r") as f:
            for line in f:
                yield f"data: {line.rstrip()}\n\n"
            while True:
                if f.tell() > os.path.getsize(LOG_FILE):
                    f.seek(0)
                line = f.readline()
                if line:
                    yield f"data: {line.rstrip()}\n\n"
                else:
                    time.sleep(0.5)
    return Response(generate(), mimetype="text/event-stream")

@app.route("/delete_checkpoint", methods=["POST"])
def delete_checkpoint():
    run_name   = request.form.get("run_name")
    ckpt_name  = request.form.get("checkpoint_name")
    path       = os.path.join(BASE_OUTPUT_DIR, run_name, ckpt_name)
    if run_name and ckpt_name and os.path.isdir(path):
        shutil.rmtree(path)
        flash(f"🗑️ Checkpoint deleted: {run_name}/{ckpt_name}")
    else:
        flash(f"❌ Could not delete checkpoint: {run_name}/{ckpt_name}")
    return redirect(url_for("index"))

@app.route("/delete_run", methods=["POST"])
def delete_run():
    run_name = request.form.get("run_name")
    path = os.path.join(BASE_OUTPUT_DIR, run_name)
    if run_name and os.path.isdir(path):
        shutil.rmtree(path)
        flash(f"🗑️ Run deleted: {run_name}")
    else:
        flash(f"❌ Could not delete run: {run_name}")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9999, debug=False)
