import sys
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from data import DataContainer
from utils import set_random_seed


def main(args):
    # data preparing
    if args.data == 'eicu':
        data_file = './data/eicu_cohort_modify.csv'
        feats_to_encode = ['labname']
        feats_to_drop = ['labtime', 'patientunitstayid']
        feats_to_scale = ['age', 'weight', 'height', 'labresult']
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
    elif args.data == 'framingham':
        data_file = './data/framingham.csv'
        feats_to_encode = []
        feats_to_drop = []
        feats_to_scale = ['age', 'education', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
        label_feature = 'TenYearCHD'
        identity_or_embedding = [True, False, False, True, False, True, True, True, True, False, False, False, False,
                                 False, False]
        origin_dims = [2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]
        target_dims = [2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]
    elif args.data == 'wgs':
        data_file = './data/wgs_dataset.csv'
        feats_to_encode = []
        feats_to_drop = []
        feats_to_scale = []
        label_feature = 'Y'
        identity_or_embedding = [True for _ in range(1884)]
        origin_dims = [3 for _ in range(1884)]
        target_dims = [3 for _ in range(1884)]
    elif args.data == 'wgs_dummy':
        data_file = './data/dummy_wgs_dataset.csv'
        feats_to_encode = []
        feats_to_drop = []
        feats_to_scale = []
        label_feature = 'Y'
        identity_or_embedding = [True for _ in range(3675)]
        origin_dims = [3 for _ in range(3675)]
        target_dims = [3 for _ in range(3675)]
    elif args.data == 'new_correct_dummy_wgs':
        data_file = './data/new_correct_wgs_dummy_dataset.csv'
        feats_to_encode = []
        feats_to_drop = []
        feats_to_scale = []
        label_feature = 'Y'
        origin_dims = [1 for _ in range(3676)]
        target_dims = [1 for _ in range(3676)]
    else:
        print('Not implement yet')

    dataObj = DataContainer(data_name=args.data, data_file=data_file, train_ratio=args.train_prop,
                            test_ratio=args.test_prop,
                            label_feat=label_feature, feats_to_encode=feats_to_encode, feats_to_drop=feats_to_drop,
                            feats_to_scale=feats_to_scale, seed=args.random_seed)

    clean_data_ = dataObj.clean()  # (n_samples, 1+feats)
    # # remove the first column in clean_data_ dataframe, and return clean_data
    # clean_data = clean_data_.iloc[:, 1:]  # TODO: 临时代码，需要整合到DataContainer中

    trainX, trainY, testX, testY, \
    _, _, hd0x_test, hd1x_test, _, _, hd0y_test, hd1y_test = dataObj.split(clean_data_)

    # Define the models
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(solver="liblinear", random_state=args.random_seed)
        ),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=args.random_seed),
        "SVM": Pipeline([('scaler', StandardScaler()), ('svc', SVC(probability=True, random_state=args.random_seed))]),
    }

    num_features = trainX.shape[1]
    if args.data == 'mimic':
        feature_names = ['Gender', 'Age', 'Weight', 'Height', 'BMI_Group', 'Charttime', 'Valuenum']
    elif args.data == 'framingham':
        feature_names = ['Male', 'Age', 'Education', 'Current Smoker', 'Cigs Per Day', 'BPMeds', 'Prevalent Stroke',
                         'Prevalent Hyp', 'Diabetes', 'Tot Chol', 'Sys BP', 'Dia BP', 'BMI', 'Heart Rate', 'Glucose']
    else:
        feature_names = [f'feat_{i + 1}' for i in range(num_features)]

    for name, model in models.items():
        model.fit(trainX, trainY.ravel())
        y_proba = model.predict_proba(testX)[:, 1]
        fpr, tpr, _ = roc_curve(testY, y_proba)
        auc_score = roc_auc_score(testY, y_proba)
        print(f"{name}: AUC = {auc_score:.4f}")

        # Find the top-3 most influential features for each model individually
        if name == "Logistic Regression":
            importances = model.named_steps['logisticregression'].coef_[0]
        elif name == "Random Forest":
            importances = model.feature_importances_
        elif name == "SVM":
            X_transformed = model.named_steps['scaler'].transform(trainX)
            result = permutation_importance(model.named_steps['svc'], X_transformed, trainY.ravel(), n_repeats=50,
                                            random_state=args.random_seed, n_jobs=-1)
            importances = result.importances_mean

        top_3_features_idx = np.argsort(importances)[-3:][::-1]
        top_3_features = [(idx, importances[idx]) for idx in top_3_features_idx]

        print(f"\n{name} - Top 3 most influential features:")
        for idx, importance in top_3_features:
            print(f"{feature_names[idx]}: {importance:.4f}")
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIME explanation for model's prediction")
    parser.add_argument("--data", required=True,
                        choices=['eicu', 'mimic', 'framingham', 'wgs', 'wgs_dummy', 'new_correct_dummy_wgs'],
                        help='Benchmark data set to run')
    parser.add_argument("--train_prop", type=float, default=0.7, help='Proportion of data to use for training')
    parser.add_argument("--test_prop", type=float, default=0.3, help='Proportion of data to use for testing')
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed for sample selection")
    args = parser.parse_args()

    set_random_seed(args.random_seed)

    main(args)