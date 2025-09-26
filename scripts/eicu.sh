cd ..

python main.py --aug False --device cpu --data eicu --train_prop 0.03 --test_prop 0.01 --batch_size 128 --epoch 500 --lr 1e-4 \
--base_tau 0.8 --tau 0.1 --hid_dim 140 --feat_hid_dim 256 --down_hid_dim 16 --dropout 0.3 --seed 4