import os
import pandas as pd
import json
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from scipy.stats import pearsonr
from scipy.stats import ttest_ind

import matplotlib.pyplot as plt
import seaborn as sns

def load_dataframe(file_name):
  return pd.read_csv(file_name)

def filter_by_mapper_node(graph_file, df, node_id):
  map_node_id = list(graph_file['nodes'].keys())
  node = graph_file['nodes'][map_node_id[node_id-1]]['vertices']

  row_ids = []
  for i in node:
    row_ids.append(i)

  return df.iloc[row_ids]




def extract_bin_intervals(df, num_intervals, perc_overlap):
    n_cubes = np.array([num_intervals])
    perc_overlap = np.array([perc_overlap / 100])
    # Extract labels
    # print("n_cubes:", n_cubes)
    # print("perc_overlap:", perc_overlap)

    labels = df['label'].values

    # Apply PCA (reduce to 1D)
    feature_cols = [x for x in df.columns.to_list() if x not in ['bin', 'second_bin', 'label', 'cluster']]
    pca = PCA(n_components=1)
    pca_data = pca.fit_transform(df[feature_cols].values).flatten()  # Convert to 1D array
    bounds = (np.array([np.min(pca_data)]), np.array([np.max(pca_data)]))


    ranges = bounds[1] - bounds[0]
    # (n-1)/n |range|
    inner_range = ((n_cubes - 1) / n_cubes) * ranges
    inset = (ranges - inner_range) / 2

    # |range| / (2n ( 1 - p))
    radius = ranges / (2 * ((n_cubes) - (n_cubes - 1) * perc_overlap))

    zip_items = list(bounds)  # work around 2.7,3.4 weird behavior
    zip_items.extend([n_cubes, radius])

    centers = [
        np.linspace(b + r, c - r, num=n) for b, c, n, r in zip(*zip_items)
    ]

    bins = []
    for center in centers[0]:
      start = center - radius
      end = center + radius
      bins.append((start[0], end[0]))

    W = pca.components_.T.flatten()

    return bins, W



def project_data_to_pca_axis(df, weight):
  feature_cols = [x for x in df.columns.to_list() if x not in ['bin', 'second_bin', 'label', 'cluster']]
  X_centered = df[feature_cols].values - np.mean(df[feature_cols].values, axis=0)
  xw = np.dot(X_centered, weight)
  final_df = pd.DataFrame(xw, columns=['PC1'])
  # print(final_df.shape, df['label'].shape)
  final_df['label'] = df['label'].reset_index(drop=True)
  return final_df

def assign_discovery_bins_to_dataframe(discovery_df, replication_df, num_intervals, perc_overlap, clip):
    # Extract labels
    bins, pca_weight = extract_bin_intervals(discovery_df, num_intervals, perc_overlap)

    labels = discovery_df['label'].values

    projected_replication_df = project_data_to_pca_axis(replication_df, pca_weight)
    projected_replication = projected_replication_df['PC1'].values
    bins_labels = np.full(len(projected_replication), -1)  # Default to -1 (no bin found)
    bins_labels_2 = np.full(len(projected_replication), -1)


    if(clip==True):
      projected_replication = np.clip(projected_replication, bins[0][0], bins[-1][1]-0.0000001)

    for i, value in enumerate(projected_replication):
        for bin_idx, (start, end) in enumerate(bins):
            # print(start, end, value)
            if start <= value and  value < end:  # Assign data point to the correct bin
                if(bins_labels[i] == -1):
                  bins_labels[i] = bin_idx + 1
                else:
                  bins_labels_2[i] = bin_idx + 1

    # print("shape1:", replication_df.shape)
    # replication_df = replication_df.reset_index(drop=True)
    # print("shape2:", replication_df.shape)
    replication_df['bin'] = bins_labels
    replication_df['second_bin'] = bins_labels_2

    # replication_df = replication_df[replication_df['bin'] != -1]
    # print("shape3:", replication_df.shape)
    # replication_df = replication_df.reset_index(drop=True)
    return replication_df


def dbscan_predict(db, X_new):
    """
    Assign cluster labels to new data points X_new using a DBSCAN model 'db'
    that was trained on some other data.

    Since DBSCAN does not have a built-in predict() method, this function
    uses the trained core samples. Points within the eps-neighborhood of a core
    sample are assigned the corresponding cluster label.

    Parameters:
        db (DBSCAN): A fitted DBSCAN instance.
        X_new (array-like or DataFrame): New data points.

    Returns:
        np.array: Cluster labels for the new points. Points not close to any core point
                  are labeled as noise (-1).
    """
    # Default: mark all as noise (-1)
    labels = np.full(shape=len(X_new), fill_value=-1, dtype=int)

    if len(db.components_) == 0:  # In case all training points were noise.
        return labels

    # Compute pairwise distances between each new point and DBSCAN's core samples.
    distances = pairwise_distances(X_new, db.components_)

    # Get the closest core point for each new point.
    closest_core = np.argmin(distances, axis=1)
    closest_distance = distances[np.arange(len(X_new)), closest_core]

    # Assign cluster label if within eps.
    for i, (dist, core_idx) in enumerate(zip(closest_distance, closest_core)):
        if dist <= db.eps:
            labels[i] = db.labels_[db.core_sample_indices_[core_idx]]
    return labels


def fit_dbscan_models(df1, eps=0.5, min_samples=5):
    """
    Trains a DBSCAN model for each unique bin in df1 and stores relevant info.

    Parameters:
    - df1: pandas.DataFrame — training dataframe; must have a 'bin' column.
    - eps: float — the eps parameter for DBSCAN.
    - min_samples: int — the min_samples parameter for DBSCAN.

    Returns:
    - A dictionary mapping each bin value to a dictionary with:
        - 'model': the trained DBSCAN model,
        - 'features': the DataFrame used for training,
        - 'labels': the resulting cluster labels from DBSCAN.
    """
    feature_cols = [x for x in df1.columns if x not in ['bin', 'second_bin', 'label', 'cluster', 'second_cluster']]
    models = {}

    for b in df1['bin'].unique():
        bin_mask = (df1['bin'] == b ) | (df1['second_bin'] == b)
        X = df1.loc[bin_mask, feature_cols].copy()
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        models[b] = {
            'model': db,
            'features': X,
            'labels': db.labels_
        }

    return models



def assign_discovery_clusters_to_dataframe(df, models, mode='replication'):
    """
    Assign cluster labels to a dataframe either by direct matching (discovery mode)
    or by prediction using dbscan_predict (replication mode).

    Parameters:
        df (pd.DataFrame): Data with 'bin' and optionally 'second_bin'.
        models (dict): Dictionary where each bin maps to:
            {
                'model': trained DBSCAN model,
                'features': DataFrame used during training,
                'labels': labels assigned during training
            }
        mode (str): Either 'discovery' or 'replication'.

    Returns:
        pd.DataFrame: Modified dataframe with 'cluster' and 'second_cluster'.
    """
    df = df.copy()
    df['cluster'] = -1
    df['second_cluster'] = -1

    feature_cols = [col for col in df.columns if col not in ['bin', 'second_bin', 'label', 'cluster', 'second_cluster']]

    for b, entry in models.items():
        model = entry['model']
        features = entry['features']
        labels = entry['labels']

        bin_mask = ((df['bin'] == b) | (df['second_bin'] == b))
        bin_indices = df[bin_mask].index

        if(mode == 'discovery'):

          for i, idx in enumerate(bin_indices):
              if df.at[idx, 'bin'] == b:
                  df.at[idx, 'cluster'] = labels[i]
              if df.at[idx, 'second_bin'] == b:
                  df.at[idx, 'second_cluster'] = labels[i]
        else:
          X_new = df.loc[bin_mask, feature_cols]
          if len(X_new) == 0:
            continue
          labels = dbscan_predict(model, X_new)
          bin_indices = df[bin_mask].index
          for i, idx in enumerate(bin_indices):
              if df.at[idx, 'bin'] == b:
                  df.at[idx, 'cluster'] = labels[i]
              if df.at[idx, 'second_bin'] == b:
                  df.at[idx, 'second_cluster'] = labels[i]
    return df

def t_test_asd_vs_adhd(df, node_df, feature):

  group1 = node_df[node_df['label'] == 'ASD'][feature].values
  group2 = node_df[node_df['label'] == 'ADHD'][feature].values

  t_stat, p_value = ttest_ind(group1, group2)

  return (feature, t_stat, p_value)


def get_shared_features_intermediate_nodes(df, graph_file, node_ids):
  shared_features = []

  progress = 0
  for i, col in enumerate(df.columns):
    if progress % 100 == 0:
      print(f"get_shared_features_intermediate_nodes - Progress: {progress}/{df.shape[1]-1}")
    progress += 1

    if col not in ['bin', 'second_bin', 'label', 'cluster', 'second_cluster']:
      red_flag = 0
      for node_id in node_ids:
        node_df = filter_by_mapper_node(graph_file, df, node_id)
        col, t_stat, p_val_t = t_test_asd_vs_adhd(df, node_df, col)
        # col, f_stat, p_val_f = f_test_asd_vs_adhd(df, node_df, col)

        if (p_val_t <= 0.5):
          red_flag = 1
          break
      if red_flag == 0:
        shared_features.append((col, t_stat, p_val_t))

  return shared_features


def open_mapper_graph(path):
  with open(path, 'r') as file:
    graph_file = json.load(file)
    return graph_file

def features_correlation_with_pca(df, pca_df, disorder=None):
  corr_p_features = []
  if disorder != None:
    df = df[df['label'] == disorder]
    pca_df = pca_df[pca_df['label'] == disorder]

  progress = 0
  for i, col in enumerate(df.columns):
    if progress % 100 == 0:
      print(f"features_correlation_with_pca ({disorder})- Progress: {progress}/{df.shape[1]-1}")
    progress += 1

    if col not in ['bin', 'second_bin', 'label', 'cluster', 'second_cluster']:
      x = df[col].values
      y = pca_df['PC1'].values
      # print(x.shape, y.shape)
      correlation, p_value = pearsonr(x, y)
      if(p_value < 0.05 and np.abs(correlation) > 0.15):
        corr_p_features.append((col, correlation, p_value))
  corr_p_features = sorted(corr_p_features, key=lambda x: x[2])
  return corr_p_features

def feature_to_regions(feature, map_df):
  row, col = index_to_row_col(feature)
  regions = (row + 1, col + 1)
  return regions

def index_to_row_col(flat_index):
    """
    Convert a flattened upper triangular index (excluding diagonal) to a row and column in the original matrix.

    Args:
    - flat_index (int): Index in the flattened upper triangular array.
    - matrix_size (int): Size of the square matrix (e.g., 190 for a 190x190 matrix).

    Returns:
    - tuple: (row, col) corresponding to the original matrix.
    """
    flat_index = int(flat_index)
    matrix_size = 190
    # Calculate the row using the inverse sum formula
    row = int(np.floor((2 * matrix_size - 1 - np.sqrt((2 * matrix_size - 1)**2 - 8 * flat_index)) / 2))

    # Calculate the column based on the row
    col = flat_index - (row * (2 * matrix_size - row - 1)) // 2 + row + 1

    return row, col

def get_mapper_features(df, graph_file, inter_nodes):

  feature_cols = [x for x in df.columns.to_list() if x not in ['bin', 'second_bin', 'label', 'cluster', 'second_cluster']]
  pca = PCA(n_components=1)
  pca_data = pca.fit_transform(df[feature_cols].values)
  pca_df = pd.DataFrame(pca_data, columns=['PC1'])
  pca_df['label'] = df['label'].values

  asd_corr_p_features = features_correlation_with_pca(df, pca_df, "ASD")
  adhd_corr_p_features = features_correlation_with_pca(df, pca_df, "ADHD")
  overall_corr_p_features = features_correlation_with_pca(df, pca_df)

  asd_features = set()
  adhd_features = set()

  feature_corr = dict()

  # return asd_corr_p_features, adhd_corr_p_features, overall_corr_p_features

  for feature, correlation, p_value in asd_corr_p_features:
    # asd_features.add(feature)
    feature_corr[feature] = [correlation, None, None]

  for feature, correlation, p_value in adhd_corr_p_features:
    if feature not in feature_corr:
      feature_corr[feature] = [None, correlation, None]
    else:
      feature_corr[feature][1] = correlation

  for feature, correlation, p_value in overall_corr_p_features:
    if feature in feature_corr:
      feature_corr[feature][2] = correlation

  significant_features = list(feature_corr.keys())


  sf = np.array(significant_features)
  print(f"Significant features: {sf.shape[0]}")


  filt_df = df[sf]
  filt_df['label'] = df['label'].values

  mapper_features = get_shared_features_intermediate_nodes(filt_df, graph_file, inter_nodes)
  
  mapper_feaures = [feature for feature, t_score, p_value in mapper_features]

  filt_feature_corr = dict()
  for feature in feature_corr.keys():
    if feature in mapper_feaures:
      filt_feature_corr[feature] = feature_corr[feature]

  mapper_features_without_none = dict()
  for feature in filt_feature_corr:
    if filt_feature_corr[feature][0] != None and filt_feature_corr[feature][1] != None:
      mapper_features_without_none[feature] = filt_feature_corr[feature]
  print(f"Shared features: {len(mapper_features_without_none.keys())}")
  return mapper_features_without_none


def multinodes_paired_boxplot(df, feature, node_ids, title):

    bp_df = pd.DataFrame(columns=['node_id', 'target_feature', 'label'])

    new_ids = []
    data = []
    labels = []
    # print("TEST1")
    for node_id in node_ids:
        # print("Node:", node_id)
        filt_df = filter_by_mapper_node(graph_file, df, node_id)
        # print("filtered")
        new_ids.extend([node_id for _ in range(len(filt_df))])
        data.extend(filt_df[feature].values)
        labels.extend(filt_df['label'].values)

    # print("TEST2")
    scaler = StandardScaler()
    data = scaler.fit_transform(np.array(data).reshape(-1, 1))

    bp_df['node_id'] = new_ids
    bp_df['node_id'] = pd.Categorical(bp_df['node_id'], categories=np.unique(node_ids), ordered=True)

    bp_df['target_feature'] = data
    bp_df['label'] = labels

    fig, ax = plt.subplots(figsize=(12, 6))

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    custom_palette = {"ADHD": default_colors[0], "ASD": default_colors[1]}  # Blue and orange

    sns.boxplot(
        y='target_feature',
        x='node_id',
        data=bp_df,
        order=node_ids,
        hue='label',
        ax=ax,
        showcaps=False,
        showfliers=False,
        boxprops={'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 2},
        whiskerprops={'color': 'black', 'linewidth': 2},
        medianprops={'color': 'red', 'linewidth': 2},
        palette=custom_palette
    )

    # Add data points behind the transparent boxes without adding them to the legend
    sns.stripplot(
        y='target_feature',
        x='node_id',
        data=bp_df,
        order=node_ids,
        hue='label',
        dodge=True,
        ax=ax,
        alpha=0.7,
        marker='o',    # Circle shape
        edgecolor='black',
        jitter=True,
        legend=False,  # Suppress legend for the dots
        linewidth=0.5,  # Outline thickness
         size=7        # Dot size (resize)
    )

    ax.set_ylabel("Activation (z-scored)", fontsize=20)
    ax.set_xlabel("Node no.", fontsize=20)
    # Adjust the legend to only reflect the boxplot
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = list(dict(zip(labels, handles)).keys())  # Unique labels for boxplot
    ax.legend(handles[:len(unique_labels)], labels[:len(unique_labels)], title='Label', loc='upper right')

    # Adjust aesthetics
    ax.set_title(title, fontsize=18, x=0.25, y=1)
    ax.set_ylim(-4,4)
    ax.tick_params(axis='both', labelsize=20)

    # ax.set_yticklabels([i for i in range(-3, 4)])
    # ax.set_xticklabels(ax.get_xticks(), fontsize=13)
    # ax.set_yticklabels(ax.get_yticks(), fontsize=13)

    return fig, ax


def multinodes_paired_boxplot_validation(df, feature, node_ids, title):
    # print("x")
    df = df[df['cluster'] != -1]

    bp_df = pd.DataFrame(columns=['node_id', 'target_feature', 'label'])


    # new_ids = []
    data = df[feature].values
    # labels = []

    # for node_id in node_ids:
    #     filt_df = filter_by_mapper_node(graph_file, df, node_id)
    #     new_ids.extend([node_id for _ in range(len(filt_df))])
    #     data.extend(filt_df[feature].values)
    #     labels.extend(filt_df['label'].values)

    scaler = StandardScaler()
    data = scaler.fit_transform(np.array(data).reshape(-1, 1))


    bp_df['node_id'] = df['bin'].values

    bp_df['target_feature'] = data
    bp_df['label'] = df['label'].values
    bp_df = bp_df[bp_df['node_id']  != -1]

    # print(bp_df.shape)
    fig, ax = plt.subplots(figsize=(12, 6))

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    custom_palette = {"ADHD": default_colors[0], "ASD": default_colors[1]}  # Blue and orange

    sns.boxplot(
        y='target_feature',
        x='node_id',
        data=bp_df,
        order=node_ids,
        hue='label',
        ax=ax,
        showcaps=False,
        showfliers=False,
        boxprops={'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 2},
        whiskerprops={'color': 'black', 'linewidth': 2},
        medianprops={'color': 'red', 'linewidth': 2},
        palette=custom_palette
    )

    # Add data points behind the transparent boxes without adding them to the legend
    sns.stripplot(
        y='target_feature',
        x='node_id',
        data=bp_df,
        order=node_ids,
        hue='label',
        dodge=True,
        ax=ax,
        alpha=0.7,
        marker='o',    # Circle shape
        edgecolor='black',
        jitter=True,
        legend=False,  # Suppress legend for the dots
        linewidth=0.5,  # Outline thickness
        size=7        # Dot size (resize)
    )
    # print("node ids=", node_ids)
    # print("Values=",np.unique(bp_df['node_id'].values))
    ax.set_ylabel("Activation (z-scored)", fontsize=20)
    ax.set_xlabel("Node no.", fontsize=20)
    # Adjust the legend to only reflect the boxplot
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = list(dict(zip(labels, handles)).keys())  # Unique labels for boxplot
    ax.legend(handles[:len(unique_labels)], labels[:len(unique_labels)], title='Label', loc='upper right')

    # ax.set_xticks(node_ids)
    # Adjust aesthetics
    ax.set_title(title)

    # ax.set_xticklabels(ax.get_xticks(), fontsize=13)
    # ax.set_yticklabels(ax.get_yticks(), fontsize=13)
    ax.set_title(title, fontsize=18, x=0.25, y=1)
    ax.set_ylim(-4,4)
    ax.tick_params(axis='both', labelsize=20)

    return fig, ax

def get_positive_both(mapper_features):
  positive_both = dict()
  for feature in mapper_features:
    if mapper_features[feature][0] > 0 and mapper_features[feature][1] > 0 and mapper_features[feature][2] != None:
      positive_both[feature] = mapper_features[feature]

  sorted_positive_both = dict(
      sorted(positive_both.items(),
             key=lambda item: (item[1][2]),
             reverse=True)
  )
  return sorted_positive_both

def get_negative_both(mapper_features):
  negative_both = dict()
  for feature in mapper_features:
    if mapper_features[feature][0] < 0 and mapper_features[feature][1] < 0 and mapper_features[feature][2] != None:
      negative_both[feature] = mapper_features[feature]
  sorted_negative_both = dict(
      sorted(negative_both.items(),
             key=lambda item: (item[1][2]),
             reverse=False)
  )
  return sorted_negative_both

def get_positive_left_only(mapper_feature):
  positive_left_only = dict()
  for feature in mapper_feature:
    if mapper_feature[feature][0] > 0 and mapper_feature[feature][1] < 0:
      positive_left_only[feature] = mapper_feature[feature]
  sorted_positive_left_only = dict(
      sorted(positive_left_only.items(),
             key=lambda item: (abs(item[1][0]) + abs(item[1][1])) / 2,
             reverse=True)
  )
  return sorted_positive_left_only

def get_positive_right_only(mapper_feature):
  positive_right_only = dict()
  for feature in mapper_feature:
    if mapper_feature[feature][0] < 0 and mapper_feature[feature][1] > 0:
      positive_right_only[feature] = mapper_feature[feature]
  sorted_positive_right_only = dict(
      sorted(positive_right_only.items(),
             key=lambda item: (abs(item[1][0]) + abs(item[1][1])) / 2,
             reverse=True)
  )
  return sorted_positive_right_only
# def get_negative_both(mapper_features):


def convert_mapper_features_to_json_serializable(mapper_features):
    """Ensure mapper_features is JSON serializable."""
    json_serializable = {}
    for key, value in mapper_features.items():
        # Convert each element of value to native Python float or list
        json_serializable[key] = [float(x) if x is not None else None for x in value]
    return json_serializable

def load_mapper_features_from_json(file_path):
    with open(file_path, 'r') as f:
        mapper_features = json.load(f)
    # Convert keys back to strings (JSON already uses strings)
    return mapper_features



def evaluate_similarty_by_feature(discovery_df, replication_df, feature, node_ids):
  p_vals = {node_id: None for node_id in node_ids}
  replication_df = replication_df[replication_df['cluster'] != -1]
  discovery_df = discovery_df[discovery_df['cluster'] != -1]
  for node_id in node_ids:
    discovery_node_df = discovery_df[((discovery_df['bin'] == node_id) & (discovery_df['cluster'] != -1)) | ((discovery_df['second_bin'] == node_id) & (discovery_df['second_cluster'] != -1))]
    replication_node_df = replication_df[((replication_df['bin'] == node_id) & (replication_df['cluster'] != -1)) | ((replication_df['second_bin'] == node_id) & (replication_df['second_cluster'] != -1))]


    if(replication_node_df.shape[0] > 0 and discovery_node_df.shape[0] > 0):
      t_test, p_value = ttest_ind(discovery_node_df[feature].values, replication_node_df[feature].values)
      p_vals[node_id] = p_value
  return p_vals



def stability_of_mapper_features(discovery_df, replication_df, mapper_features):

  stability_dict = dict()
  not_none_nodes = []
  progress = 0
  for feature in mapper_features:
    p_vals = evaluate_similarty_by_feature(discovery_df, replication_df, feature, [i for i in range(1,20)])
    stability_dict[feature] = 0
    count_not_none = 0
    count_pass = 0
    for node_id in p_vals.keys():
      if p_vals[node_id] != None:
        # if(node_id not in not_none_nodes):
        #   not_none_nodes.append(node_id)
        count_not_none += 1
        if p_vals[node_id] >= 0.05:
          count_pass += 1
    progress += 1
    if(progress % 100 == 0):
      print(f"stability_of_mapper_features - Progress: {progress}/{len(mapper_features)}")

    # print(count_pass)
    # print(count_not_none)
    # print(not_none_nodes)
    stability_dict[feature] = count_pass / count_not_none


  return stability_dict

def average_stability(discovery_df, replication_df, features):
  stability_dict = stability_of_mapper_features(discovery_df, replication_df, features)
  count = 0
  sum = 0
  for feature in stability_dict:
    sum += stability_dict[feature]
    count += 1
  avg = sum / count
  return avg

def process_pattern(pattern_func, pattern_name, mapper_features, map_df, discovery_df, replication_df, max_features=11):
    print(f"\nProcessing pattern: {pattern_name}")
    pattern = pattern_func(mapper_features)

    # Plot discovery
    for n, feature in enumerate(pattern.keys()):
        if n >= max_features:
            break
        regions = feature_to_regions(feature, map_df)
        title = f"Region {regions[0]} - Region {regions[1]} (discovery)"
        fig, ax = multinodes_paired_boxplot(discovery_df, feature, range(1, 20), title)
        file_name = f"region_{regions[0]}-{regions[1]}_{pattern_name}_discovery.png"
        fig.savefig(f"plots/{file_name}", dpi=300, bbox_inches='tight')
        plt.close()

    # Plot replication
    for n, feature in enumerate(pattern.keys()):
        if n >= max_features:
            break
        regions = feature_to_regions(feature, map_df)
        title = f"Region {regions[0]} - Region {regions[1]} (replication)"
        fig, ax = multinodes_paired_boxplot_validation(replication_df, feature, range(1, 20), title)
        file_name = f"region_{regions[0]}-{regions[1]}_{pattern_name}_replication.png"
        fig.savefig(f"plots/{file_name}", dpi=300, bbox_inches='tight')
        plt.close()

    # Stability analysis
    stability_dict = stability_of_mapper_features(discovery_df, replication_df, list(pattern.keys())[:5])
    avg_stability = average_stability(discovery_df, replication_df, list(pattern.keys()))
    print(f"{pattern_name} average stability: {avg_stability}")
    return pattern


if __name__ == "__main__":
    GRAPH_DIR = "MapperInteractive/app/static/uploads"

    discovery_df = load_dataframe('data/discovery_df.csv')
    replication_df = load_dataframe('data/replication_df.csv')
    
    graph_file = open_mapper_graph(f"{GRAPH_DIR}/discovery_df.csv_dbscan_eps_32_minpts_5_filter_PC1/mapper_discovery_df_674.csv_20_40.json")
    map_df = pd.read_excel(f"data/mapping_from_190_to_200_ROIs.xlsx")

    discovery_df = assign_discovery_bins_to_dataframe(discovery_df, discovery_df, 20, 40, clip=True)
    replication_df = assign_discovery_bins_to_dataframe(discovery_df, replication_df, 20, 40, clip=False)
    
    
    models = fit_dbscan_models(discovery_df, 32, 5)
    discovery_df = assign_discovery_clusters_to_dataframe(discovery_df, models, "discovery")
    replication_df = assign_discovery_clusters_to_dataframe(replication_df, models)
    

    FEATURES_FILE = "data/mapper_features.json"
    
    if os.path.exists(FEATURES_FILE):
        print("Loading precomputed mapper features from file...")
        mapper_features = load_mapper_features_from_json(FEATURES_FILE)
    else:
        print("Computing mapper features...")
        mapper_features = get_mapper_features(discovery_df, graph_file, [11])
        # Convert to JSON serializable before saving
        mapper_features_serializable = convert_mapper_features_to_json_serializable(mapper_features)
        with open(FEATURES_FILE, "w") as f:
            json.dump(mapper_features_serializable, f, indent=4)
        print(f"Saved mapper features to {FEATURES_FILE}")


    print("Number of transition features:", len(mapper_features))

    first_pattern = process_pattern(get_positive_both, "first_pattern", mapper_features, map_df, discovery_df, replication_df)
    second_pattern = process_pattern(get_negative_both, "second_pattern", mapper_features, map_df, discovery_df, replication_df)
    third_pattern = process_pattern(get_positive_left_only, "third_pattern", mapper_features, map_df, discovery_df, replication_df)
    fourth_pattern = process_pattern(get_positive_right_only, "fourth_pattern", mapper_features, map_df, discovery_df, replication_df)
    # first_pattern = get_positive_both(mapper_features)
    # second_pattern = get_negative_both(mapper_features)
    # third_pattern = get_positive_left_only(mapper_features)
    # fourth_pattern = get_positive_right_only(mapper_features)

    # for n, feature in enumerate(first_pattern.keys()):
    #   if(n < 11):
        
    #     regions = feature_to_regions(feature, map_df)
    #     title = f"Region {regions[0]} - Region {regions[1]} (discovery)"
    #     fig, ax = multinodes_paired_boxplot(discovery_df, feature, [i for i in range(1,20)], title)
    #     file_name = f"region_{regions[0]}-{regions[1]}_discovery.png"
    #     fig.savefig(f"plots/{file_name}", dpi=300, bbox_inches='tight')
    #     plt.close()

    # for n, feature in enumerate(first_pattern.keys()):
    #   if(n < 11):
    #     regions = feature_to_regions(feature, map_df)
    #     title = f"Region {regions[0]} - Region {regions[1]} (replication)"
    #     fig, ax = multinodes_paired_boxplot_validation(replication_df, feature, [i for i in range(1,20)], title)
    #     file_name = f"region_{regions[0]}-{regions[1]}_replication.png"
    #     fig.savefig(f"plots/{file_name}", dpi=300, bbox_inches='tight')
    #     plt.close()

    # stability_dict = stability_of_mapper_features(discovery_df, replication_df, list(first_pattern.keys())[:5])
    # avg_stability = average_stability(discovery_df, replication_df, list(first_pattern.keys()))

    # print(avg_stability)