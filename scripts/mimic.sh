cd ..

#python main.py --aug True --aug_size 2048 --n_component 3 --device cpu --data mimic --train_prop 0.3 --test_prop 0.3 --batch_size 256 --epoch 500 --lr 5e-4 \
#--base_tau 0.8 --tau 0.1 --hid_dim 140 --feat_hid_dim 210 --down_hid_dim 16 --dropout 0.3 --seed 0

# full: 0.7-0.3; Medium: 0.5-0.3; Small: 0.3-0.3
python main.py --data mimic --device cpu --train_prop 0.7 --test_prop 0.3 \
--aug True  --n_component 1 --aug_size 128 \
--batch_size 128 --epoch 1000 --lr 5e-4 \
--base_tau 0.5 --tau 0.9 --hid_dim 140 --feat_hid_dim 210 --down_hid_dim 16 --dropout 0.3 --seed 2 \
--patch 1 --num_block 1