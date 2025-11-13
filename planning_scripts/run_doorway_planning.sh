n_trials=100
buffer=0.01
safe_time=0.1
video=""

python planning/DualArm_puzzle_planning.py  --experiment_name experiment_mc_seed0 --save_traj --num_trials $n_trials --buffer $buffer --safe_time $safe_time --provide_initial_condition $video --consider_manifold_constraint
python planning/DualArm_puzzle_planning.py  --planner_mode simple --save_traj --num_trials $n_trials $video --buffer $buffer --safe_time $safe_time # simple planner

