#!/bin/bash
#SBATCH --account=mth250011p
#SBATCH --job-name=final_project_reg_finetuned_bert
#SBATCH --cpus-per-task=5
#SBATCH --time=06:00:00
#SBATCH -o ../logs/reg_finetuned_bert.out
#SBATCH -e ../logs/reg_finetuned_bert.err
#SBATCH --partition=GPU-shared 
#SBATCH --gpus=1
#SBATCH --mem=16G     

echo "=== Running Ridge Regression on Finetuned BERT embeddings ==="
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 16G"

module load anaconda3
source ~/.bashrc
conda activate env_215a

python run_regression.py

conda deactivate
echo "=== Job completed ==="