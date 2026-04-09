# Gemma-4 E4B
Gemma-4 model for solving long homophonic substitution ciphers

## Usage AI-lab
### Initialization
- `uv sync`

### Training
- `sbatch train.slurm`

#### Train without spaces
- `sbatch train.slurm --without-spaces`

### Monitoring during training
- `tail -f logs/train_live_<JOB_ID>.log`

### Cancellation of training
- `scancel --name=gemma_4_finetune`
- `scancel -u USERNAME`

## Usage UCloud
1. Setup a container with pytorch version 25.11 on UCloud
2. Upload the train.sh to UCloud
3. Use train.sh as the batch script to run when the container is available

## Configuration
All parameters are listed in `src/config.py`.