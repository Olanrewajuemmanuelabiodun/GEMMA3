# GEMMA3 (SLM) - Build and Inference

Minimal, notebook-driven project to build a Small Language Model (SLM) from scratch and run inference.

- Training notebook: `Gemma3.ipynb`
- Inference notebook: `Inference.ipynb`
- Asset: `15SA-3FA.png`
- Example validation split: `validation.bin.zip` (unzips to `validation.bin`)

## What this repo does
- Loads the `roneneldan/TinyStories` dataset via `datasets`.
- Tokenizes with `tiktoken` (GPT-2 encoding) and writes memory-mapped binaries: `train.bin` and `validation.bin`.
- Prepares input/output batches for training.
- Defines a compact transformer SLM (with RoPE).
- Provides a separate notebook to run inference once weights are trained/loaded.

## Requirements
- Python 3.10+
- Recommended: NVIDIA GPU + CUDA for faster training

Install packages:
```bash
pip install torch datasets tiktoken numpy pandas tqdm pyarrow huggingface-hub
```

Notes:
- Hugging Face access may show an auth warning; public datasets usually work without a token. If needed, set `HF_TOKEN`.
- Google Colab works well; the notebooks include lightweight install cells.

## Quickstart
1) Open `Gemma3.ipynb` and run cells top-to-bottom to:
- Install dependencies
- Download `TinyStories`
- Tokenize to `train.bin` and `validation.bin`
- Configure hyperparameters and train the SLM

2) Open `Inference.ipynb` to:
- Load the trained weights/checkpoint
- Generate text from prompts

## Hugging Face model repository (pretrained weights)
A ready-to-use repository exists on the Hugging Face Hub:
- Repository: [`EmmanuelOlanrewaju/gemma3-model`](https://huggingface.co/EmmanuelOlanrewaju/gemma3-model)

Contents (as of latest update):
- `.gitattributes`
- `config.json` (model configuration)
- `modeling_gemma3.py` (model architecture code)
- `best_model_params-2.pt` (~258 MB; PyTorch weights)

You can use these files to run inference without re-training locally.

### Download options

1) Python API (recommended)
```python
# pip install huggingface-hub
from huggingface_hub import hf_hub_download

# Download checkpoint and config to a local cache and return file paths
ckpt_path = hf_hub_download(repo_id="EmmanuelOlanrewaju/gemma3-model", filename="best_model_params-2.pt")
config_path = hf_hub_download(repo_id="EmmanuelOlanrewaju/gemma3-model", filename="config.json")
code_path = hf_hub_download(repo_id="EmmanuelOlanrewaju/gemma3-model", filename="modeling_gemma3.py")
print(ckpt_path, config_path, code_path)
```

2) Command line (no auth required for public repos)
```bash
# Option A: clone the repo with git-lfs
# If needed: brew install git-lfs && git lfs install

git clone https://huggingface.co/EmmanuelOlanrewaju/gemma3-model

# Option B: use the huggingface-cli to download specific files
# pip install huggingface-hub
huggingface-cli download EmmanuelOlanrewaju/gemma3-model best_model_params-2.pt --local-dir ./gemma3-model
huggingface-cli download EmmanuelOlanrewaju/gemma3-model config.json --local-dir ./gemma3-model
huggingface-cli download EmmanuelOlanrewaju/gemma3-model modeling_gemma3.py --local-dir ./gemma3-model
```

3) In Google Colab
```python
!pip install -q huggingface_hub
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download("EmmanuelOlanrewaju/gemma3-model", "best_model_params-2.pt")
```

### Using the Hub files in this project
Place the downloaded files where your notebooks can find them (e.g., project root or a subdirectory like `./checkpoints`). Then, in `Inference.ipynb`, update the paths accordingly.

Generic loading pattern:
```python
import json, torch, importlib.util, types, os

# Paths to your local copies (adjust as needed)
ckpt_path = "./gemma3-model/best_model_params-2.pt"
config_path = "./gemma3-model/config.json"
model_code_path = "./gemma3-model/modeling_gemma3.py"

# 1) Load config
with open(config_path, "r") as f:
    config = json.load(f)

# 2) Dynamically import the model code from modeling_gemma3.py
spec = importlib.util.spec_from_file_location("modeling_gemma3", model_code_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)  # module now contains the model class/factory

# 3) Construct the model; check modeling_gemma3.py for the exact entry point
# Examples (choose the one that matches your file):
# model = module.Gemma3Model(**config)
# model = module.build_model(config)
# model = module.create_model(config)

# 4) Load the state dict
state = torch.load(ckpt_path, map_location="cpu")
# Some checkpoints save as {"model_state_dict": ..., ...}; adjust if needed
state_dict = state.get("model_state_dict", state)
# model.load_state_dict(state_dict, strict=False)

# 5) Switch to eval and generate
# model.eval()
# with torch.no_grad():
#     output_ids = model.generate(input_ids, max_new_tokens=100, temperature=0.8)
```

Notes:
- If your `modeling_gemma3.py` exports a different construction API, open the file and adjust the call accordingly.
- If you trained on GPU and load on CPU, use `map_location="cpu"`.
- If key names do not match exactly, try `strict=False` when calling `load_state_dict`.

### Integrating with the notebooks
- `Gemma3.ipynb`: you can skip tokenization/training and jump to evaluation cells by downloading weights from the Hub first.
- `Inference.ipynb`: set the checkpoint and config paths to the downloaded files and run generation.

## Data preparation (headless)
```python
from datasets import load_dataset
import tiktoken, numpy as np, os

ds = load_dataset("roneneldan/TinyStories")
# Tokenize with GPT-2 encoding and write np.memmap files (see notebook for full code)
```
This produces `train.bin` and `validation.bin` used by training.

## Repository structure
- `Gemma3.ipynb`: training, tokenization, batching, model definition
- `Inference.ipynb`: load checkpoint and run generation
- `validation.bin.zip`: zipped validation split (unzip before use)
- `15SA-3FA.png`: supporting image asset

## Tips
- Ensure enough disk space for memmaps.
- Use a GPU runtime for speed (e.g., Colab: Runtime > Change runtime type > GPU).

## License
Add a `LICENSE` file (e.g., MIT) if you plan to distribute.

## Acknowledgements
- Inspired by nanoGPT utilities
- Uses `roneneldan/TinyStories` and `tiktoken`
