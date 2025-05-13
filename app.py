# app.py
import sys
import os
import json
import re
import time
from flask import (
    Flask, request, render_template,
    jsonify, Response, redirect, url_for
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

# ensure folders exist
os.makedirs("datasets", exist_ok=True)
os.makedirs("logs", exist_ok=True)
OUTPUT_DIR = "result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Default config (OUTPUT_DIR fixed) ---
available_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
default_config = {
    "MODEL_NAME": "Qwen/Qwen2.5-0.5B-Instruct",
    "BATCH_SIZE": 1,
    "EPOCHS": 10,
    "LEARNING_RATE": 5e-5,
    "DATA_PATH": "",  # will set to first existing dataset if any
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
    "save_strategy": "epoch",
    "save_total_limit": 5,
    "cuda_devices": ",".join(available_devices),
}
config = default_config.copy()

# If there's at least one file, default DATA_PATH
datasets_list = sorted(os.listdir("datasets"))
if datasets_list:
    config["DATA_PATH"] = os.path.join("datasets", datasets_list[0])

# --- Log & status files ---
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

    write_status("Training in progress...")
    try:
        write_log("=== Training started ===\n")

        write_log(f"Loading dataset from {config['DATA_PATH']}...\n")
        ds = load_dataset("json", data_files=config["DATA_PATH"])
        write_log("Dataset loaded.\n")

        write_log("Initializing tokenizer & model...\n")
        kwargs = {}
        if "gemma" in config["MODEL_NAME"]:
            kwargs["attn_implementation"] = "eager"
        if len(config["cuda_devices"].split(",")) > 1:
            kwargs["device_map"] = "auto"
        tokenizer = AutoTokenizer.from_pretrained(config["MODEL_NAME"], **kwargs)
        model     = AutoModelForCausalLM.from_pretrained(config["MODEL_NAME"], **kwargs)
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
                json.dumps(ex["messages"]),
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
            learning_rate=config["LEARNING_RATE"],
            per_device_train_batch_size=config["BATCH_SIZE"],
            per_device_eval_batch_size=config["BATCH_SIZE"],
            num_train_epochs=config["EPOCHS"],
            weight_decay=config["weight_decay"],
            logging_dir=config["logging_dir"],
            logging_steps=config["logging_steps"],
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
        write_status("Completed")
    except Exception as e:
        write_log(f"ERROR: {e}\n")
        write_status("Failed")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

@app.route("/", methods=["GET"])
def index():
    # refresh datasets & models
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
    return redirect(url_for("index"))

@app.route("/train", methods=["POST"])
def train():
    # update config from form
    for k in config:
        if k in request.form and request.form[k]:
            v = request.form[k]
            try:    config[k] = int(v)
            except:
                try:    config[k] = float(v)
                except: config[k] = v

    # ensure DATA_PATH uses selected dataset
    if "DATA_PATH" in request.form:
        config["DATA_PATH"] = request.form["DATA_PATH"]

    # start training
    global training_process
    if training_process and training_process.is_alive():
        return jsonify(status="Already running"), 400

    open(LOG_FILE, "w").close()
    write_status("Queued")

    training_process = Process(target=train_model)
    training_process.start()
    return jsonify(status="Training started")

@app.route("/stop", methods=["POST"])
def stop():
    global training_process
    if training_process and training_process.is_alive():
        training_process.terminate()
        write_status("Stopped")
        return jsonify(status="Force killed"), 200
    return jsonify(status="No active training"), 400

@app.route("/clear_logs", methods=["POST"])
def clear_logs():
    open(LOG_FILE, "w").close()
    return jsonify(status="Logs cleared")

@app.route("/status")
def status():
    st = "Idle"
    try:
        st = json.load(open(STATUS_FILE)).get("status", st)
    except:
        pass
    return jsonify(status=st)

@app.route("/logs")
def logs():
    def generate():
        open(LOG_FILE, "a").close()
        with open(LOG_FILE) as f:
            for line in f:
                yield f"data: {line.rstrip()}\n\n"
            while True:
                line = f.readline()
                if line:
                    yield f"data: {line.rstrip()}\n\n"
                else:
                    time.sleep(0.5)
    return Response(generate(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9999, debug=True)
