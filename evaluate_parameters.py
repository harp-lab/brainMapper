import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

def load_dataframe(file_name):
  return pd.read_csv(file_name)

def plot_projection_function(df, proj_f):
  fig, ax = plt.subplots(figsize=(8, 6))

  if proj_f == "PC1":
    pca = PCA(n_components=1)
    pca_data = pca.fit_transform(df.iloc[:, :-1].values)
    x = pca_data

  elif proj_f == "l1-norm":
    l1norm = np.linalg.norm(df.iloc[:, :-1].values, ord=1, axis=1)
    x = l1norm

  elif proj_f == "l2-norm":
    l2norm = np.linalg.norm(df.iloc[:, :-1].values, ord=2, axis=1)
    x = l2norm

  labels = df.iloc[:, -1].values  # Labels
  # Encode the labels to numeric values
  encoded_labels = LabelEncoder().fit_transform(labels)

  # Separate x values by class
  class_0_x = x[encoded_labels == 0]  # x values for class 0
  class_1_x = x[encoded_labels == 1]  # x values for class 1

  # Get the original class labels (before encoding)
  original_labels = np.unique(labels)

  # Plot histograms for each class
  ax.hist(class_0_x, bins=20, alpha=0.7, label=original_labels[0], color='#1f77b4', edgecolor='black')
  ax.hist(class_1_x, bins=20, alpha=0.7, label=original_labels[1], color='#ff7f0e', edgecolor='black')

  # Add labels and title
  ax.set_xlabel(proj_f, fontsize=20)
  ax.set_ylabel('Frequency', fontsize=20)
  ax.tick_params(axis='both', labelsize=16)


  # Add legend with original string labels
  ax.legend(fontsize=20)

  # plt.savefig(f"{proj_f}.png")
  fig.savefig(f"plots/{proj_f}.png", dpi=300, bbox_inches='tight')
  plt.close(fig)
  # Show plot
  # plt.show()




def elbow_methods(df, k):
  fig, ax = plt.subplots(figsize=(8, 6))
  feature_columns = [col for col in df.columns if col not in ['label', 'bin', 'second_bin', 'cluster', 'second_cluster']]
  data = df[feature_columns].values
  neighbors = NearestNeighbors(n_neighbors=k)
  neighbors_fit = neighbors.fit(data)
  distances, indices = neighbors_fit.kneighbors(data)

  k_distances = np.sort(distances[:, k-1])
  ax.plot(k_distances, linewidth=5)
  ax.set_xlabel("Sorted points", fontsize=20)
  ax.set_ylabel(f"{k}-NN distance", fontsize=20)
  # plt.title("k-distance Graph (Elbow Method)")
  ax.tick_params(axis='both', labelsize=20)
  ax.grid(True)
  fig.savefig(f"plots/elbow.png", dpi=300, bbox_inches='tight')
  plt.close()


if __name__ == "__main__":

    discovery_df = load_dataframe('data/discovery_df.csv')
    replication_df = load_dataframe('data/replication_df.csv')

    plot_projection_function(discovery_df, "PC1")
    plot_projection_function(discovery_df, "l1-norm")
    plot_projection_function(discovery_df, "l2-norm")
    elbow_methods(discovery_df, 5)



    



