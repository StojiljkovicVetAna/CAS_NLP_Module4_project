#!/bin/bash
#SBATCH --account=XXX
#SBATCH --time=08:00:00
#SBATCH --wckey=XXX
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:h200:1
#SBATCH --job-name="apertus-epoch1"
#SBATCH --output=logs/apertus_epoch1_%j.log

echo "========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================="

# Load Anaconda and activate conda environment
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate CAS_NLP_M4

# Verify GPU availability
nvidia-smi

# Train epoch 1 (from scratch)
python train_apertus_epoch.py \
    --model_name "swiss-ai/Apertus-8B-2509" \
    --output_dir "models/apertus_fairytales" \
    --checkpoint_dir "" \
    --num_epochs 1 \
    --epoch_num 1 \
    --batch_size 1 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 5

echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
