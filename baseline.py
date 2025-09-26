import sys
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from data import DataContainer

# Parse arguments
parser = argparse.ArgumentParser(description='Run Baseline Models')
parser.add_argument("--data", required=True, choices=['eicu', 'mimic', 'framingham', 'wgs_group1', 'wgs_group2', 'wgs_group3'], help='Benchmark data set to run')
parser.add_argument("--train_prop", type=float, default=0.7, help='Proportion of data to use for training')
parser.add_argument("--test_prop", type=float, default=0.3, help='Proportion of data to use for testing')
parser.add_argument("--seed", default=0, type=int, help='')
args = parser.parse_args()

# Data configurations from main.py
if args.data == 'eicu':
    data_file = './data/eicu_cohort_modify.csv'
    feats_to_encode = ['labname']
    feats_to_drop = ['labtime','patientunitstayid']
    feats_to_scale = ['age','weight','height','labresult']
    label_feature = 'aki_stage'
elif args.data == 'mimic':
    data_file = './data/mimic.csv'
    feats_to_encode = []
    feats_to_drop = ['itemid', 'icustay_id', 'charttime', 'valuenum', 'age', 'gender', 'mean_weight']
    feats_to_scale = ['bmi_group', 'height']
    label_feature = 'aki_stage'
elif args.data == 'framingham':
    data_file = './data/framingham.csv'
    feats_to_encode = []
    feats_to_drop = []
    feats_to_scale = ['age','education','cigsPerDay','totChol','sysBP','diaBP','BMI','heartRate','glucose']
    label_feature = 'TenYearCHD'
elif args.data == 'wgs_group1':
    data_file = './data/new_correct_wgs_dummy_dataset.csv'
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

# Load and preprocess data
clean_data_ = dataObj.clean()
trainX, trainY, testX, testY, _, _, _, _, _, _, _, _ = dataObj.split(clean_data_)

print('dataset loaded.')

# Initialize models
logistic_regression = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=args.seed))
random_forest = RandomForestClassifier(random_state=args.seed)
svm_model = make_pipeline(StandardScaler(), SVC(probability=True, random_state=args.seed))

# Train and evaluate models
models = {
'Logistic Regression': logistic_regression,
'Random Forest': random_forest,
'SVM': svm_model
}

for name, model in models.items():
    model.fit(trainX, trainY.ravel())
    y_proba = model.predict_proba(testX)[:, 1]
    fpr, tpr, _ = roc_curve(testY, y_proba)
    auc_score = roc_auc_score(testY, y_proba)
    print(f"{name}: AUC = {auc_score:.4f}")