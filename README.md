# LLM Web GUI Trainer Dashboard

This Flask application provides a web-based dashboard to configure, manage, and monitor the training of language models (e.g., Qwen) with optional LoRA fine-tuning. It supports uploading and deleting datasets, selecting which dataset to train on, real-time log streaming, and model management.

## Features

* **Interactive Configuration**: Set hyperparameters such as batch size, epochs, learning rate, LoRA settings, padding strategy, and device selection.
* **Dataset Management**:

  * Upload new datasets (JSON format) via the **Datasets** tab.
  * Delete existing datasets.
  * Select which dataset to use for training in the **Configuration** tab.
* **Training Control**:

  * Start and stop training runs from the dashboard.
  * Status indicators with emojis for clear feedback (e.g., 🚀, ✅, ❌).
  * Fixed `DATA_PATH` that points to the selected dataset and cannot be arbitrarily changed in code.
* **Log Streaming**:

  * Real-time server-sent events (SSE) log streaming to the **Logs** tab.
  * Automatic handling of log file truncation between runs.
  * Clear logs with a single button click.
* **Model Management**:

  * View existing trained model directories under `result/`.
* **Flash Messages**:

  * Instant feedback on uploads, deletions, and other actions via Bootstrap alerts.

## Installation

1. **Clone the repository**:

   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```
2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate   # Windows
   ```
3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
├── app.py                # Main Flask application
├── templates/
│   └── index.html        # Dashboard UI
├── datasets/             # Uploaded dataset files
├── logs/
│   ├── flask_logs.txt    # Consolidated log file
│   └── status.json       # JSON file with current training status
├── result/               # Output directory for trained models and logs
└── requirements.txt      # Python dependencies
```

## Usage

1. **Run the app**:

   ```bash
   export FLASK_APP=app.py
   export FLASK_ENV=development  # optional
   flask run --host=0.0.0.0 --port=9999
   ```
2. **Open in browser**: Navigate to `http://localhost:9999`.
3. **Upload datasets** under the **Datasets** tab (JSON files with `messages` arrays).
4. **Configure** hyperparameters and select a dataset in the **Configuration** tab.
5. **Start Training** with the ▶️ button. Monitor progress in **Logs**.
6. **Stop Training** anytime with the ⏹️ button.
7. **View trained models** under **Models**.

## Environment Variables

* `SECRET_KEY`: Flask secret key for session flashing (default: `supersecretkey`).

## Customization

* To change default hyperparameters, edit the `default_config` dictionary in `app.py`.
* LoRA settings can be toggled via `use_lora` and adjusted with `lora_r`, `lora_alpha`, `lora_dropout`.
* Device selection is automatic but can be customized via `cuda_devices`.

## Contributing

Contributions are welcome! Please open issues or pull requests for improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
