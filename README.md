# PSRO4math

## Datasets construction

Ensure you are in folder PSRO-datasets.


### Step 1

Please run `python download_aops_instruct.py` to download the AOPS Instruct dataset from [here](https://huggingface.co/datasets/DeepStudentLlama/AoPS-Instruct).

### Step 2

Fill the `API_BASE`,`API_KEY` and `CHAT_COMPLETIONS_PATH` in `generate_qwen3_proofs.py`,`oracle_label_gpt5_inplace.py` and `meta_verification_inplace.py`.


Please run `python run_all.py` to automatically constrcuct the datasets.