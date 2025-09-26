import sys
import argparse
import torch
import numpy as np

from data import DataContainer, augmentation
from model import HybridLT
from trainer import Trainer
from utils import plot_roc_curve, set_random_seed

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

# Parse Arguments
parser = argparse.ArgumentParser(description='Run IDEM HybridLT Model')
# general hyperparams
parser.add_argument("--device", type=str, default='cpu', help='')
parser.add_argument("--seed", default=0, type=int, help='')

# dataset-related
parser.add_argument("--data", required=True, choices=['eicu', 'mimic', 'framingham', 'wgs_group1', 'wgs_group2', 'wgs_group3'], help='Benchmark data set to run')
parser.add_argument("--train_prop", type=float, default=0.7, help='Proportion of data to use for training')
parser.add_argument("--test_prop", type=float, default=0.3, help='Proportion of data to use for testing')
parser.add_argument("--aug", default=True, type=str_to_bool, help='')
parser.add_argument("--n_component", type=int, default=1, help='')
parser.add_argument("--aug_size", type=int, default=256, help='')
# training-related
parser.add_argument("--epoch", type=int, default=500, help='Number of epochs for which to train NN models')
parser.add_argument("--batch_size", type=int, default=512, help='Size of batches for NN models')
parser.add_argument("--lr", type=float, default=1e-3, help='')
parser.add_argument("--weight_decay", type=float, default=1e-5, help='')
# model-related
parser.add_argument("--hid_dim", type=int, default=64, help='')
parser.add_argument("--feat_hid_dim", type=int, default=64, help='')
parser.add_argument("--down_hid_dim", type=int, default=16, help='')
parser.add_argument("--base_tau", type=float, default=0.07, help='')
parser.add_argument("--tau", type=float, default=0.07, help='')
parser.add_argument("--dropout", type=float, default=0.2, help='')
parser.add_argument("--patch", type=int, default=1, help='')
parser.add_argument("--num_block", type=int, default=1, help='')

args = parser.parse_args()

set_random_seed(args.seed)

device = torch.device(args.device)

# data preparing
if args.data == 'eicu':
    data_file = './data/eicu_cohort_modify.csv'
    feats_to_encode = ['labname']
    feats_to_drop = ['labtime','patientunitstayid']
    feats_to_scale = ['age','weight','height','labresult']
    label_feature = 'aki_stage'
    identity_or_embedding = [True, False, False, False, True, True, False]
    origin_dims = [2, 1, 1, 1, 5, 10, 1]
    target_dims = [2, 1, 1, 1, 5, 10, 1]
elif args.data == 'mimic':
    data_file = './data/mimic.csv'
    feats_to_encode = []
    feats_to_drop = ['itemid', 'icustay_id']
    feats_to_scale = ['age', 'mean_weight', 'height', 'charttime', 'valuenum']
    label_feature = 'aki_stage'
    identity_or_embedding = [True, False, False, False, True, False, False]
    origin_dims = [2, 1, 1, 1, 4, 1, 1]
    target_dims = [2, 1, 1, 1, 4, 1, 1]
# elif args.data == 'mimic':
#     data_file = './data/mimic.csv'
#     feats_to_encode = []
#     feats_to_drop = ['itemid', 'icustay_id', 'charttime', 'valuenum', 'age', 'gender', 'mean_weight']
#     feats_to_scale = ['bmi_group', 'height']
#     label_feature = 'aki_stage'
#     identity_or_embedding = [True, False]
#     origin_dims = [4, 1]
#     target_dims = [4, 1]
elif args.data == 'framingham':
    data_file = './data/framingham.csv'
    feats_to_encode = []
    feats_to_drop = []
    feats_to_scale = ['age','education','cigsPerDay','totChol','sysBP','diaBP','BMI','heartRate','glucose']
    label_feature = 'TenYearCHD'
    identity_or_embedding = [True, False, False, True, False, True, True, True, True, False, False, False, False, False, False]
    origin_dims = [2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]
    target_dims = [2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]
# elif args.data == 'wgs':
#     data_file = './data/wgs_dataset.csv'
#     feats_to_encode = []
#     feats_to_drop = []
#     feats_to_scale = []
#     label_feature = 'Y'
#     identity_or_embedding = [True for _ in range(1884)]
#     origin_dims = [3 for _ in range(1884)]
#     target_dims = [3 for _ in range(1884)]
# elif args.data == 'wgs_dummy':
#     data_file = './data/dummy_wgs_dataset.csv'
#     feats_to_encode = []
#     feats_to_drop = []
#     feats_to_scale = []
#     label_feature = 'Y'
#     identity_or_embedding = [True for _ in range(3675)]
#     origin_dims = [3 for _ in range(3675)]
#     target_dims = [3 for _ in range(3675)]
elif args.data == 'wgs_group1':
    data_file = './data/wgs_dummy_dataset.csv'
    feats_to_encode = []
    feats_to_drop = []
    feats_to_scale = []
    label_feature = 'Y'
    origin_dims = [1 for _ in range(3676)]
    target_dims = [1 for _ in range(3676)]
elif args.data == 'wgs_group2':
    data_file = './data/group2_dummy_dataset.csv'
    feats_to_encode = []
    feats_to_drop = []
    feats_to_scale = []
    label_feature = 'Y'
    origin_dims = [1 for _ in range(6055)]
    target_dims = [1 for _ in range(6055)]
elif args.data == 'wgs_group3':
    data_file = './data/group3_dummy_dataset.csv'
    feats_to_encode = []
    feats_to_drop = []
    feats_to_scale = []
    label_feature = 'Y'
    origin_dims = [1 for _ in range(14402)]
    target_dims = [1 for _ in range(14402)]
else:
    print('Not implement yet')

dataObj = DataContainer(data_name=args.data, data_file=data_file, train_ratio=args.train_prop, test_ratio=args.test_prop,
                        label_feat=label_feature, feats_to_encode=feats_to_encode, feats_to_drop=feats_to_drop,
                        feats_to_scale=feats_to_scale, seed=args.seed)

clean_data_ = dataObj.clean()  # (n_samples, 1+feats)
# # remove the first column in clean_data_ dataframe, and return clean_data
# clean_data = clean_data_.iloc[:, 1:]  #TODO: 临时代码，需要整合到DataContainer中

trainX, trainY, testX, testY, \
_, _, hd0x_test, hd1x_test, _, _, hd0y_test, hd1y_test = dataObj.split(clean_data_)

# if args.aug:
#     trainX, trainY = augmentation(trainX, trainY, n_components=args.n_component, augmentation_size=args.aug_size)
#     np.save('trainX.npy', trainX)
#     np.save('trainY.npy', trainY)
#     print('# augmented data saved')
# else:
#     print('# loaded augmented data')
#     trainX = np.load('trainX.npy')
#     trainY = np.load('trainY.npy')

hd1x_train = trainX[trainY.reshape(-1) == 1.0]
hd0x_train = trainX[trainY.reshape(-1) == 0.0]
hd1y_train = trainY[trainY.reshape(-1) == 1.0].reshape(-1)
hd0y_train = trainY[trainY.reshape(-1) == 0.0].reshape(-1)

nn_data = trainX, trainY, testX, testY, \
          hd0x_train, hd1x_train, hd0x_test, hd1x_test, hd0y_train, hd1y_train, hd0y_test, hd1y_test

# log data statistics
print('train samples: {}, test samples: {}'.format(len(trainX), len(testX)))
print('number of 1-class train samples: {}, and the ratio is: {}'.format(len(hd1x_train), len(hd1x_train) / len(trainX)))
print('number of 0-class train samples: {}, and the ratio is: {}'.format(len(hd0x_train), len(hd0x_train) / len(trainX)))
print('number of 1-class test samples: {}, and the ratio is: {}'.format(len(hd1x_test), len(hd1x_test) / len(testX)))
print('number of 0-class test samples: {}, and the ratio is: {}'.format(len(hd0x_test), len(hd0x_test) / len(testX)))

# initialize model and trainer
model = HybridLT(inp_dim=trainX.shape[-1], hid_dim=args.hid_dim, feat_learn_dim=args.feat_hid_dim,
                 downstream_dim=args.down_hid_dim, base_tau=args.base_tau, tau=args.tau, dropout=0.2, device=device,
                 patch=args.patch, block=args.num_block).to(device)

engine = Trainer(model=model, data=nn_data, epochs=args.epoch, batch_size=args.batch_size,
                 lr=args.lr, weight_decay=args.weight_decay, device=device)

# train & test model
engine.train()
model_fpr, model_tpr, model_auc = engine.evaluate(engine.best_model)

# save model
torch.save(engine.best_model, 'best_model.pt')

# plot_roc_curve(model_fpr, model_tpr, model_auc, title='HybridLT', savefig=None, show=True)