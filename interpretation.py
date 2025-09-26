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
    return torch.load('./pretrained/mimic_best_model_seed3.pt', map_location=torch.device('cpu'))


def test_model(model, testX, testY):
    """
    Test the model accuracy on the test set.

    Parameters:
    -----------
    model : pytorch model object
        The pre-trained model object.
    testX : numpy ndarray
        The feature matrix of test set.
    testY : numpy ndarray
        The label array of test set.

    Returns:
    --------
    test_acc : float
        The test set accuracy of the model.
    """
    model.eval()
    with torch.no_grad():
        testX_tensor = torch.tensor(testX)
        testY_tensor = torch.tensor(testY).squeeze()
        preds = model.predict(testX_tensor).squeeze()
        test_acc = torch.mean((preds > 0.5).float().eq(testY_tensor.float()).float()).item()
    print(f"Test accuracy: {test_acc}")
    return test_acc


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
    # # remove the first column in clean_data_ dataframe, and return clean_data
    # clean_data = clean_data_.iloc[:, 1:]  # TODO: 临时代码，需要整合到DataContainer中

    trainX, trainY, testX, testY, \
    _, _, hd0x_test, hd1x_test, _, _, hd0y_test, hd1y_test = dataObj.split(clean_data_)

    model = load_model()
    model.eval()
    classifier = predict_instance(model)

    # Initialize LIME explainer
    num_features = trainX.shape[1]
    if args.data == 'mimic':
        feature_names = ['Gender', 'Age', 'Weight', 'Height', 'BMI_Group', 'Charttime', 'Valuenum']
    else:
        feature_names = [f'feat_{i+1}' for i in range(num_features)]
    explainer = CustomLimeTabularExplainer(trainX, feature_names=feature_names, class_names=['Class 0', 'Class 1'],
                                           verbose=True,
                                           mode='classification')

    # print model accuracy on test set
    test_acc = test_model(model, testX, testY)

    # Randomly select samples
    random_samples = np.random.choice(len(testX), size=args.num_samples, replace=False)

    top_k_features = 20

    # Analyze and visualize the selected samples
    for i, sample_idx in enumerate(random_samples):
        instance = testX[sample_idx]
        true_label = testY[sample_idx][0]
        predicted_label = predict_instance_label(instance, model)

        exp = explainer.explain_instance(instance, classifier, num_features=top_k_features, true_label=true_label)

        print(f"Sample {i+1}:")
        print(f"Predicted Label: {predicted_label}, True Label: {true_label}")
        print("Saving feature importances:")
        file_path = f"lime_explanation_sample_{i}.html"
        with open(file_path, 'w') as file_:
            html_content = exp.as_html(labels=(1,), show_predicted_value=True)

            # Find the position to insert the title and script
            insert_index = html_content.find('<div id="lime-container"')

            # Create the title and style elements
            title_html = f'<div id="true-label-title" style="position: absolute; top: 0; left: 0; padding: 10px;"><h3>True Label: {true_label}</h3></div>'

            # Insert the title and style into the HTML content
            html_content = (
                    html_content[:insert_index]
                    + title_html
                    + html_content[insert_index:]
            )

            file_.write(html_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIME explanation for model's prediction")
    parser.add_argument("--data", required=True,
                        choices=['eicu', 'mimic', 'framingham', 'wgs', 'wgs_dummy', 'new_correct_dummy_wgs'],
                        help='Benchmark data set to run')
    parser.add_argument("--train_prop", type=float, default=0.7, help='Proportion of data to use for training')
    parser.add_argument("--test_prop", type=float, default=0.3, help='Proportion of data to use for testing')
    parser.add_argument("--num_samples", type=int, help="Number of randomly selected samples to be tested")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed for sample selection")
    args = parser.parse_args()

    set_random_seed(args.random_seed)

    main(args)