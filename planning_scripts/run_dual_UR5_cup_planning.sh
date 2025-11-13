n_trials=100
video=""
safe_time=0.3
buffer=0.05

python planning/DualArm_cup_planning.py --experiment_name experiment_mc_seed0 --num_trials $n_trials --buffer $buffer --safe_time $safe_time --save_traj --provide_initial_condition  --consider_manifold_constraint $video
python planning/DualArm_cup_planning.py --num_trials $n_trials --planner_mode simple  --save_traj  $video