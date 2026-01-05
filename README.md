# PSRO4math

A dataset construction pipeline for math proof verification, integrated with ROLL (Reinforcement Learning Optimization for Large-Scale Learning) for model training.

## Project Structure

```
PSRO4math/
├── PSRO-datasets/           # Dataset construction pipeline
│   ├── download_aops_instruct.py
│   ├── generate_proofs_vllm_offline.py
│   ├── oracle_label_inplace.py
│   ├── convert_proof_gen_sft.py
│   └── ...
└── ROLL/                    # RL framework (git submodule)
```

## Setup

```bash
# Clone with submodule
git clone --recursive https://github.com/tanzelin430/PSRO4math.git

# Or initialize submodule after clone
git submodule update --init --recursive

# Install ROLL
cd ROLL && pip install -e .
```

## Dataset Construction

All commands run from `PSRO-datasets/` directory.

### Step 1: Download Base Dataset

```bash
python download_aops_instruct.py
```

Downloads AoPS-Instruct dataset from [HuggingFace](https://huggingface.co/datasets/DeepStudentLlama/AoPS-Instruct).

### Step 2: Generate Proofs

**Option A: Local vLLM (recommended)**

```bash
# Configure MODEL_NAME and TENSOR_PARALLEL_SIZE in the script
python generate_proofs_vllm_offline.py
```

**Option B: Remote API**

```bash
# Configure API_BASE, API_KEY in the script
python generate_qwen3_proofs.py
```

### Step 3: Oracle Labeling

```bash
# Configure verifier API settings
python oracle_label_inplace.py
```

### Step 4: Convert to SFT Format

```bash
python convert_proof_gen_sft.py    # For proof generation SFT
python convert_to_sft_format.py    # For scoring model SFT
```

## SFT Training with ROLL

See `ROLL/examples/` for training configurations:

- `qwen3-4B-proof-gen/` - Proof generation model
- `qwen3-4B-sft_scoring/` - Proof scoring model

```bash
cd ROLL
python examples/start_sft_pipeline.py \
  --config_path qwen3-4B-proof-gen \
  --config_name sft_config
```

## Pipeline Flow

```
AoPS-Instruct (CSV)
    ↓ generate_proofs_vllm_offline.py
Candidate Proofs (JSONL)
    ↓ oracle_label_inplace.py
Labeled Proofs with GT scores (JSONL)
    ↓ convert_*_sft.py
SFT Training Data (JSON)
    ↓ ROLL SFT Pipeline
Trained Model
```

## Scoring System

- **1.0**: Completely correct proof
- **0.5**: Minor errors or omissions
- **0.0**: Incorrect or fatal errors

## License

MIT
