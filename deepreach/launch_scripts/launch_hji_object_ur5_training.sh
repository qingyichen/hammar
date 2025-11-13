seeds=(0)
for seed in "${seeds[@]}";do
    CUDA_VISIBLE_DEVICES=0 python deepreach/experiment_scripts/train_hji_object_ur5.py --experiment_name experiment_mc --minWith zero --tMax 0.5 --num_src_samples 10000 --pretrain --pretrain_iters 10000 --num_epochs 120000 --counter_end 110000 --seed $seed --consider_constraint_manifold
done
