import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import umap

from utils import set_random_seed
from data import DataContainer, augmentation


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

    # combine trainX and testX, as well as trainY and testY
    X = np.concatenate([trainX, testX], axis=0)
    Y = np.concatenate([trainY, testY], axis=0)
    ALL = np.concatenate([X, Y], axis=1)

    print(X.shape)
    print(Y.shape)

    # calculate z-scores
    z_scores = np.abs(stats.zscore(X))

    # set threshold to remove outliers
    threshold = 3
    # get indices of outliers
    outliers = np.where(z_scores > threshold)
    outlier_indices = np.unique(outliers[0])

    # remove outliers from X and Y
    X_clean = np.delete(X, outlier_indices, axis=0)
    Y_clean = np.delete(Y, outlier_indices, axis=0)

    # # visualization 1. T-SNET visualization
    # tsne = TSNE(n_components=2, random_state=0)  # n_components=2 for 2D visualization
    # X_2d = tsne.fit_transform(X)
    #
    # # create a data frame containing your PCA results
    # tsne_df = pd.DataFrame(data=X_2d, columns=["comp_1", "comp_2"])
    # tsne_df["label"] = Y  # assuming Y contains your labels
    #
    # # plot the PCA results
    # plt.figure(figsize=(8,8))
    # sns.scatterplot(
    #     x="comp_1", y="comp_2",
    #     hue="label",
    #     palette=sns.color_palette("hsv", len(np.unique(Y))),
    #     data=tsne_df,
    #     legend="full",
    #     alpha=0.8
    # )
    # plt.show()

    # # visualization 2. PCA visualization
    # pca = PCA(n_components=2)  # n_components=2 for 2D visualization
    # X_2d_pca = pca.fit_transform(X_clean)
    #
    # # create a data frame containing your PCA results
    # pca_df = pd.DataFrame(data=X_2d_pca, columns=["PC1", "PC2"])
    # pca_df["label"] = Y_clean  # assuming Y contains your labels
    #
    # # plot the PCA results
    # plt.figure(figsize=(8,8))
    # sns.scatterplot(
    #     x="PC1", y="PC2",
    #     hue="label",
    #     palette=sns.color_palette("hsv", len(np.unique(Y_clean))),
    #     data=pca_df,
    #     legend="full",
    #     alpha=0.8
    # )
    # plt.show()

    # visualization 3. UMAP visualization
    label_field = ALL[:, 3].reshape(-1, 1)
    kmeans = KMeans(n_clusters=4)  # choose the number of clusters
    label_field = kmeans.fit_predict(label_field)

    reducer = umap.UMAP()
    X_2d_umap = reducer.fit_transform(ALL)

    # create a data frame containing your UMAP results
    umap_df = pd.DataFrame(data=X_2d_umap, columns=["UMAP1", "UMAP2"])
    umap_df["label"] = label_field  # assuming Y contains your labels

    # plot the UMAP results
    plt.figure(figsize=(8,8))
    sns.scatterplot(
        x="UMAP1", y="UMAP2",
        hue="label",
        palette=sns.color_palette("hsv", len(np.unique(label_field))),
        data=umap_df,
        legend="full",
        alpha=0.8
    )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data statistics visualization")
    parser.add_argument("--data", required=True,
                        choices=['eicu', 'mimic', 'framingham', 'wgs', 'wgs_dummy', 'new_correct_dummy_wgs'],
                        help='Benchmark data set to run')
    parser.add_argument("--train_prop", type=float, default=0.7, help='Proportion of data to use for training')
    parser.add_argument("--test_prop", type=float, default=0.3, help='Proportion of data to use for testing')
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed for sample selection")
    args = parser.parse_args()

    set_random_seed(args.random_seed)

    main(args)