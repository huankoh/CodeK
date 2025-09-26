import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lime import lime_tabular
from data import DataContainer, augmentation
import torch
from utils import set_random_seed


# Load your pre-trained model here
def load_model():
    # import pretrained model
    return torch.load('./pretrained/new_correct_dummy_wgs_best_model_seed0.pt', map_location=torch.device('cpu'))


def predict_instance_label(instance, model, threshold=0.5):
    instance_tensor = torch.tensor(instance).view(1, -1)
    prob = model.predict(instance_tensor).item()
    label = int(prob >= threshold)
    return label


class predict_instance():
    def __init__(self, model):
        self.model = model
    def __call__(self, instances):
        instances_tensor = torch.tensor(instances)
        instances_tensor = instances_tensor.to(self.model.backbone_encoder.embed.weight.dtype)
        probabilities = self.model.predict(instances_tensor).numpy()
        probabilities = np.column_stack((1 - probabilities, probabilities))  # Adjust the output for binary classification
        return probabilities


def get_matching_samples(testX, testY, model):
    matched_samples = []
    for i, (instance, true_label) in enumerate(zip(testX, testY)):
        predicted_label = predict_instance_label(instance, model)
        if predicted_label == true_label:
            matched_samples.append(i)
    return matched_samples


# Custom LIME explainer class to include true label in the title
class CustomLimeTabularExplainer(lime_tabular.LimeTabularExplainer):
    def explain_instance(self, data_row, predict_fn, labels=(1,), true_label=None, **kwargs):
        explanation = super().explain_instance(data_row, predict_fn, labels, **kwargs)
        if true_label is not None:
            explanation.true_label = true_label
        return explanation


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
    # remove the first column in clean_data_ dataframe, and return clean_data
    clean_data_ = clean_data_.iloc[:, 1:]  # TODO: 临时代码，需要整合到DataContainer中

    trainX, trainY, testX, testY, \
    _, _, hd0x_test, hd1x_test, _, _, hd0y_test, hd1y_test = dataObj.split(clean_data_)

    model = load_model()
    model.eval()
    classifier = predict_instance(model)

    # Initialize LIME explainer
    num_features = trainX.shape[1]
    if args.data == 'mimic':
        feature_names = ['Gender', 'Age', 'Weight', 'Height', 'BMI_Group', 'Charttime', 'Valuenum']
    elif args.data == 'framingham':
        feature_names = ['Male', 'Age', 'Education', 'Current Smoker', 'Cigs Per Day', 'BPMeds', 'Prevalent Stroke',
                         'Prevalent Hyp', 'Diabetes', 'Tot Chol', 'Sys BP', 'Dia BP', 'BMI', 'Heart Rate', 'Glucose']
    else:
        feature_names = [f'feat_{i + 1}' for i in range(num_features)]
    explainer = CustomLimeTabularExplainer(trainX, feature_names=feature_names, class_names=['Class 0', 'Class 1'],
                                           verbose=True,
                                           mode='classification')

    # Find samples where the model prediction and ground truth matched
    matching_samples = get_matching_samples(testX, testY, model)
    print(f"Number of matching samples: {len(matching_samples)}")

    feature_importances = {}

    # Analyze the matched samples
    for i, sample_idx in enumerate(matching_samples):
        instance = testX[sample_idx]
        true_label = testY[sample_idx][0]
        predicted_label = predict_instance_label(instance, model)

        exp = explainer.explain_instance(instance, classifier, num_features=len(feature_names), true_label=true_label)

        for feature, importance in exp.as_map()[1]:
            if feature not in feature_importances:
                feature_importances[feature] = [abs(importance)]
            else:
                feature_importances[feature].append(abs(importance))

    # Compute the average importance of each feature
    avg_importances = {feat: np.mean(imp_list) for feat, imp_list in feature_importances.items()}

    # Find the top-3 most influential features
    top_3_features = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)[:3]

    print("\nTop 3 most influential features for correct predictions:")
    for feature, importance in top_3_features:
        print(f"{feature_names[feature]}: {importance}")


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