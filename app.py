# app.py
import sys
import os
import json
import re
import time
from flask import (
    Flask, request, render_template,
    jsonify, Response, redirect, url_for, flash
)
from werkzeug.utils import secure_filename
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import torch
from multiprocessing import Process

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "supersecretkey")

# ensure folders exist
os.makedirs("datasets", exist_ok=True)
os.makedirs("logs", exist_ok=True)
OUTPUT_DIR = "result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

available_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
default_config = {
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "text_mapping": "messages",
    "batch_size": 1,
    "epochs": 10,
    "learning_rate": 5e-5,
    "DATA_PATH": "",  # set below based on selection
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
    "logging_dir": f"{OUTPUT_DIR}/logs",
    "logging_steps": 10,
    "logging_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 5,
    "cuda_devices": ",".join(available_devices),
}
config = default_config.copy()

# Pre-set DATA_PATH to first dataset if exists
datasets_list = sorted(os.listdir("datasets"))
if datasets_list:
    config["DATA_PATH"] = os.path.join("datasets", datasets_list[0])

LOG_FILE    = os.path.join("logs", "flask_logs.txt")
STATUS_FILE = os.path.join("logs", "status.json")
open(LOG_FILE,    "a").close()
open(STATUS_FILE, "a").close()

def write_status(s: str):
    with open(STATUS_FILE, "w") as f:
        json.dump({"status": s}, f)

original_stdout = sys.stdout
original_stderr = sys.stderr

def write_log(msg: str):
    original_stdout.write(msg)
    original_stdout.flush()
    with open(LOG_FILE, "a") as f:
        for line in re.split(r"\r\n|\r|\n", msg):
            if line.strip():
                f.write(line + "\n")

def train_model():
    class Logger:
        def write(self, m): write_log(m)
        def flush(self): pass
    sys.stdout = Logger(); sys.stderr = Logger()

    write_status("🚀 Training in progress...")
    try:
        write_log("=== Training started ===\n")

        write_log(f"Loading dataset from {config['DATA_PATH']}...\n")
        ds = load_dataset("json", data_files=config["DATA_PATH"])
        write_log("Dataset loaded.\n")

        write_log("Initializing tokenizer & model...\n")
        kwargs = {}
        if "gemma" in config["model_name"]:
            kwargs["attn_implementation"] = "eager"
        if len(config["cuda_devices"].split(",")) > 1:
            kwargs["device_map"] = "auto"
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"], **kwargs)
        model     = AutoModelForCausalLM.from_pretrained(config["model_name"], **kwargs)
        write_log("Model ready.\n")

        if config["use_lora"]:
            peft_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config["lora_r"],
                lora_alpha=config["lora_alpha"],
                lora_dropout=config["lora_dropout"],
                target_modules=["q_proj","v_proj"]
            )
            model = get_peft_model(model, peft_cfg)
            write_log("LoRA applied.\n")

        write_log("Tokenizing...\n")
        def tok(ex):
            inp = tokenizer(
                json.dumps(ex[config["text_mapping"]]),
                padding=config["padding_strategy"],
                truncation=True,
                max_length=config["padding_max_length"]
            )
            inp["labels"] = inp["input_ids"].copy()
            return inp

        tokenized = ds.map(tok, batched=False)
        splits = tokenized["train"].train_test_split(
            test_size=config["eval_dataset_percentage"]
        )
        write_log("Tokenization done.\n")

        args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            eval_strategy=config["eval_strategy"],
            learning_rate=config["learning_rate"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            num_train_epochs=config["epochs"],
            weight_decay=config["weight_decay"],
            logging_dir=config["logging_dir"],
            logging_steps=config["logging_steps"],
            logging_strategy=config["logging_strategy"],
            save_strategy=config["save_strategy"],
            save_total_limit=config["save_total_limit"],
        )
        if config["eval_strategy"] == "steps":
            args.eval_steps = config["eval_steps"]

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=splits["train"],
            eval_dataset=splits["test"],
        )
        write_log("Entering training loop...\n")
        trainer.train()
        write_log("Training loop ended.\n")

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
    models   = sorted(
        d for d in os.listdir(OUTPUT_DIR)
        if os.path.isdir(os.path.join(OUTPUT_DIR, d))
    )
    return render_template(
        "index.html",
        config=config,
        devices=available_devices,
        datasets=datasets,
        models=models
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
    # update config from form (except DATA_PATH text field)
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

    # set DATA_PATH based on dropdown
    selected = request.form.get("dataset")
    if selected:
        config["DATA_PATH"] = os.path.join("datasets", selected)

    global training_process
    if 'training_process' in globals() and training_process.is_alive():
        return jsonify(status="❌ Already running"), 400

    open(LOG_FILE, "w").close()
    write_status("⏳ Queued")

    training_process = Process(target=train_model)
    training_process.start()
    return jsonify(status="🚀 Training started")

@app.route("/stop", methods=["POST"])
def stop():
    global training_process
    if 'training_process' in globals() and training_process.is_alive():
        training_process.terminate()
        write_status("⏹️ Stopped")
        return jsonify(status="⏹️ Force killed"), 200
    return jsonify(status="❌ No active training"), 400

@app.route("/clear_logs", methods=["POST"])
def clear_logs():
    open(LOG_FILE, "w").close()
    return jsonify(status="🧹 Logs cleared")

@app.route("/status")
def status():
    st = "🤖 Idle"
    try:
        st = json.load(open(STATUS_FILE)).get("status", st)
    except:
        pass
    return jsonify(status=st)

@app.route("/logs")
def logs():
    def generate():
        # ensure file exists
        open(LOG_FILE, "a").close()

        with open(LOG_FILE, "r") as f:
            # send any existing lines

            for line in f:
                yield f"data: {line.rstrip()}\n\n"
                # now tail, but reset on truncation

            while True:
                # if file was truncated, rewind
                try:
                    if f.tell() > os.path.getsize(LOG_FILE):
                        f.seek(0)
                except FileNotFoundError:
                    # in case file was removed, recreate & rewind
                    open(LOG_FILE, "a").close()
                    f.seek(0)
                line = f.readline()
                if line:
                    yield f"data: {line.rstrip()}\n\n"
                else:
                    time.sleep(0.5)
    return Response(generate(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9999, debug=True)
