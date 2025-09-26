cd ..
# full: 0.7-0.3; Medium: 0.5-0.3; Small: 0.3-0.3
python main.py --aug True --aug_size 256 --n_component 1 --device cpu --data framingham --train_prop 0.7 --test_prop 0.3 --batch_size 512 --epoch 200 --lr 5e-4 \
--base_tau 0.8 --tau 0.1 --hid_dim 150 --feat_hid_dim 300 --down_hid_dim 16 --dropout 0.3 --patch 1 --num_block 1 --seed 2