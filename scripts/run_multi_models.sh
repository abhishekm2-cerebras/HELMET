#!/bin/bash -l

##############################
#    Multi-Model Runner      #
##############################

# Usage: bash scripts/run_multi_models.sh
# This script submits run_short_slurm.sh for each model in the arrays below

# Define model arrays - add your models here
MODEL_NAMES=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "allenai/Olmo-3-7B-Instruct"
    # "/lustre/scratch/users/ahmed.frikha/ckpts/LC_8B_SFT_baseline_251203_reasoning_added/hf/checkpoint_10816_to_hf"
    # "/lustre/scratch/users/ahmed.frikha/ckpts/LC_8B_SFT_baseline_251203_TC_3X/hf/checkpoint_10734_to_hf"
    # "/lustre/scratch/users/ahmed.frikha/ckpts/LC_8B_SFT_baseline_251213_MT_3X_mz2p7/hf/checkpoint_10915_to_hf"
    # "/lustre/scratch/users/ahmed.frikha/ckpts/LC_8B_SFT_baseline_260102_SC_3X_v2p7/hf/checkpoint_9483_to_hf"
    "/lustre/scratch/users/ahmed.frikha/ckpts/Jais-2-8B-HF_64K_yarn_8X"
    "Qwen/Qwen3-8B"
    # Add more models here
)

MNAMES=(
    "Llama-3.1-8B-Instruct"
    "Olmo-3-7B-Instruct"
    # "LC_8B_SFT_baseline_251203_reasoning_added"
    # "LC_8B_SFT_baseline_251203_TC_3X"
    # "LC_8B_SFT_baseline_251213_MT_3X_mz2p7"
    # "LC_8B_SFT_baseline_260102_SC_3X_v2p7"
    "Jais-2-8B-HF_64K_yarn_8X"
    "Qwen3-8B"  
    # Add corresponding short names here
)

# Verify arrays have same length
if [ ${#MODEL_NAMES[@]} -ne ${#MNAMES[@]} ]; then
    echo "Error: MODEL_NAMES and MNAMES arrays must have the same length"
    exit 1
fi

# Submit a job for each model
for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MNAME="${MNAMES[$i]}"

    echo "Submitting job for: $MNAME ($MODEL_NAME)"

    sbatch scripts/run_short_slurm.sh "$MODEL_NAME" "$MNAME"

    echo "Submitted job for $MNAME"
    echo "---"
done

echo "All jobs submitted!"
