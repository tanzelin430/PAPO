# PSRO4math

## Datasets construction

Ensure you are in folder PSRO-datasets.


### Step 1

Please run `python download_aops_instruct.py` to download the AOPS Instruct dataset from [here](https://huggingface.co/datasets/DeepStudentLlama/AoPS-Instruct).

### Step 2

**Option A: API-based generation**

Fill `API_BASE`, `API_KEY`, `CHAT_COMPLETIONS_PATH` in `generate_qwen3_proofs.py`, `oracle_label_gpt5_inplace.py`, `meta_verification_inplace.py`, then run `python run_all.py`.

**Option B: Local vLLM generation**

Configure `MODEL_NAME` (local model path) and `TENSOR_PARALLEL_SIZE` (number of GPUs) in `generate_proofs_vllm_offline.py`, then run:
```bash
python generate_proofs_vllm_offline.py
```