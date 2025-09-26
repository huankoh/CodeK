cd ..
# full: 0.7-0.3; Medium: 0.5-0.3; Small: 0.3-0.3
python main.py --data wgs_group2 --device cpu --train_prop 0.1 --test_prop 0.3 \
--aug True  --n_component 1 --aug_size 128 \
--batch_size 128 --epoch 1000 --lr 5e-4 \
--base_tau 0.5 --tau 0.9 --hid_dim 128 --feat_hid_dim 128 --down_hid_dim 15 --dropout 0.3 --seed 2 \
--patch 1 --num_block 1