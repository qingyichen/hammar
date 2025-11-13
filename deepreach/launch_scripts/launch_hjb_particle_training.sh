seeds=(0)
python deepreach/experiment_scripts/train_hjb_particle.py --experiment_name experiment_mc  --num_src_samples 10000 --pretrain --pretrain_iters 10000 --num_epochs 120000 --counter_end 110000 --seed 0 --consider_constraint_manifold
python deepreach/experiment_scripts/train_hjb_particle.py --experiment_name experiment_no_mc  --num_src_samples 10000 --pretrain --pretrain_iters 10000 --num_epochs 120000 --counter_end 110000 --seed 0


