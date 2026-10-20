import os
import traceback
from pathlib import Path

from datasets import load_dataset
from flask import Flask, jsonify, render_template, request
from transformers import AutoTokenizer

app = Flask(__name__)

# Base path for self_gen data
BASE_DATA_PATH = Path(__file__).parent.parent / "data" / "raw_datasets" / "self_gen"



# Cache for tokenizers
tokenizer_cache = {}


def get_tokenizer(model_path):
    """Get or create tokenizer with caching"""
    if model_path not in tokenizer_cache:
        try:
            tokenizer_cache[model_path] = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            print(f"Error loading tokenizer for {model_path}: {e}")
            return None
    return tokenizer_cache[model_path]


def discover_folders():
    """Discover all model folders in self_gen directory"""
    folders = []
    if not BASE_DATA_PATH.exists():
        return folders

    for vendor_dir in BASE_DATA_PATH.iterdir():
        if vendor_dir.is_dir():
            for model_dir in vendor_dir.iterdir():
                if model_dir.is_dir():
                    rel_path = model_dir.relative_to(BASE_DATA_PATH)
                    folders.append(str(rel_path))

    return sorted(folders)


def discover_parquet_files(folder_path):
    """Discover all parquet files in a folder"""
    # Only folders found under the data dir; this blocks "../" and absolute
    # paths without breaking symlinked data folders.
    if folder_path not in discover_folders():
        return []
    full_path = BASE_DATA_PATH / folder_path
    parquet_files = []

    if full_path.exists():
        for parquet_file in full_path.glob("**/*.parquet"):
            rel_path = parquet_file.relative_to(full_path)
            parquet_files.append(str(rel_path))

    return sorted(parquet_files)


def extract_model_name_from_folder(folder_path):
    """Extract base model name from folder path"""
    # e.g., "google/gemma-2-2b-it_temp_0.0_closed_qa_prob_1.0" -> "google/gemma-2-2b-it"
    parts = folder_path.split("/")
    if len(parts) >= 2:
        vendor = parts[0]
        model_part = parts[1].split("_temp_")[0]
        return f"{vendor}/{model_part}"
    return None


@app.route("/")
def index():
    """Main page"""
    folders = discover_folders()
    return render_template("self_gen_viewer.html", folders=folders)


@app.route("/api/folders")
def api_folders():
    """API endpoint to get available folders"""
    folders = discover_folders()
    return jsonify({"folders": folders})


@app.route("/api/parquet_files")
def api_parquet_files():
    """API endpoint to get parquet files in a folder"""
    folder = request.args.get("folder", "")
    if not folder:
        return jsonify({"error": "No folder specified"}), 400

    files = discover_parquet_files(folder)
    return jsonify({"files": files})


@app.route("/api/load_data")
def api_load_data():
    """API endpoint to load and display data from a parquet file"""
    folder = request.args.get("folder", "")
    parquet_file = request.args.get("file", "")
    try:
        num_samples = int(request.args.get("num_samples", 100))
    except ValueError:
        return jsonify({"error": "num_samples must be an integer"}), 400
    num_samples = max(1, min(num_samples, 10000))

    if not folder or not parquet_file:
        return jsonify({"error": "Missing parameters"}), 400
    # Only folders that actually exist under the data dir (also limits which
    # tokenizer names can be loaded from the Hub).
    if folder not in discover_folders():
        return jsonify({"error": "Unknown folder"}), 400

    try:
        # Construct full path
        # Only files actually listed for this folder (allowlist, no raw paths).
        if parquet_file not in discover_parquet_files(folder):
            return jsonify({"error": "File not found"}), 404
        full_path = BASE_DATA_PATH / folder / parquet_file

        # Extract model name for tokenizer
        model_name = extract_model_name_from_folder(folder)
        if not model_name:
            return jsonify({"error": "Could not extract model name from folder"}), 400

        # Load tokenizer
        tokenizer = get_tokenizer(model_name)
        if tokenizer is None:
            return jsonify({"error": f"Could not load tokenizer for {model_name}"}), 500

        # Load dataset
        ds = load_dataset(
            "parquet", data_files=str(full_path), split=f"train[:{num_samples}]"
        )

        # Process samples
        samples = []
        for i, sample in enumerate(ds):
            processed_sample = {
                "index": i,
                "ctx": tokenizer.decode(sample["ctx_ids"], skip_special_tokens=False)
                if "ctx_ids" in sample
                else "N/A",
                "questions": [],
            }

            # Decode input_ids if present
            if "input_ids" in sample:
                if isinstance(sample["input_ids"][0], list):
                    # Multiple Q&A pairs
                    processed_sample["questions"] = [
                        tokenizer.decode(qa, skip_special_tokens=False)
                        for qa in sample["input_ids"]
                    ]
                else:
                    # Single item
                    processed_sample["questions"] = [
                        tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
                    ]

            samples.append(processed_sample)

        return jsonify(
            {
                "success": True,
                "num_samples": len(samples),
                "model_name": model_name,
                "file_path": str(parquet_file),
                "samples": samples,
            }
        )

    except Exception as e:
        traceback.print_exc()
        # Details go to the server log only, not to the client.
        return jsonify({"error": f"Internal error ({type(e).__name__})"}), 500


if __name__ == "__main__":
    print(f"Data path: {BASE_DATA_PATH}")
    print(f"Available folders: {discover_folders()}")
    # Local-only by default. To expose on the network set e.g.
    # SELF_GEN_VIEWER_HOST=0.0.0.0. The Werkzeug debugger (remote code
    # execution) is never enabled for non-local hosts.
    host = os.environ.get("SELF_GEN_VIEWER_HOST", "127.0.0.1")
    port = int(os.environ.get("SELF_GEN_VIEWER_PORT", "5001"))
    debug = os.environ.get("SELF_GEN_VIEWER_DEBUG") == "1" and host in (
        "127.0.0.1",
        "localhost",
    )
    app.run(debug=debug, host=host, port=port)
