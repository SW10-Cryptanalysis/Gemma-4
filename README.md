# HF Model
Fine-tuning of HuggingFace causal language models for solving long homophonic substitution ciphers.

Supported model families: `gemma`, `llama`, `mistral`, `jamba`, `mamba`

## Usage — AI Lab (SLURM)

### Initialization
```bash
uv sync
```

### Training

Both `--model-family` and `--model-path` are **required**. The job will exit immediately with an error if either is missing.

```bash
sbatch train.slurm --model-family <family> --model-path <hf-id-or-local-path>
```

**Examples:**
```bash
# Gemma 4
sbatch train.slurm --model-family gemma --model-path google/gemma-4-E4B

# LLaMA 3.1
sbatch train.slurm --model-family llama --model-path meta-llama/Llama-3.2-3B

# Mistral
sbatch train.slurm --model-family mistral --model-path mistralai/Mistral-7B-v0.3
```

#### Optional flags
| Flag | Description |
|---|---|
| `--without-spaces` | Train without space tokens in the dataset |

```bash
sbatch train.slurm --model-family llama --model-path meta-llama/Llama-3.2-3B --without-spaces
```

### Monitoring during training
```bash
tail -f logs/train_live_<JOB_ID>.log
```

### Cancelling a job
```bash
scancel --name=hf_finetune
# or by username:
scancel -u USERNAME
```

---

## Usage — UCloud

1. Setup a container with PyTorch version 25.11 on UCloud
2. Upload `train.sh` to UCloud
3. Use `train.sh` as the batch script, passing the required flags

---

## Configuration

All parameters are listed in `src/config.py`.

Outputs are saved to `outputs/<model-family>/spaces/` or `outputs/<model-family>/normal/` depending on whether space tokens are used.