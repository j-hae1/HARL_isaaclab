for seed in {1..10}; 
do
    python train.py --algo happo --env mamujoco --exp_name Mujoco-Ant-HARL --seed $seed
done