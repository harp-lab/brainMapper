import numpy as np
import os
from data_loader import BrainDataset
import pandas as pd

def flatten_upper_triangle(matrix):
    """
    Flatten the upper triangular part of a square matrix, excluding the diagonal.

    Args:
    - matrix (np.array): A 190x190 matrix to be flattened.

    Returns:
    - np.array: Flattened 1D array of upper triangular elements, excluding the diagonal.
    """
    # Get the upper triangular indices (excluding the diagonal)
    upper_triangle = np.triu(matrix, k=1)

    # Flatten the upper triangular part into a 1D array
    return upper_triangle[upper_triangle != 0]



def convert_to_asd_adhd(df):
  asd = ['Autism', 'Asperger']
  adhd = ['ADHD-Combined', 'ADHD-Hyperactive', 'ADHD-Inattentive']
  new_labels = []
  for label in df['label']:
    if label in asd:
      new_labels.append('ASD')
    elif label in adhd:
      new_labels.append('ADHD')
  df['label'] = new_labels
  return df



def extract_site_info():
  sites = []

  site_map = {
      1 : 'PEKING',
      2 : 'BRADLEY',
      3 : 'KKI',
      4 : 'NEURO',
      5 : 'NYU',
      6 : 'OREGON',
      7 : 'PITT',
      8 : 'WU',
  }

  name = {
      'CALTECH' : 'CALTECH',
      'CMU' : 'CMU',
      'KKI' : 'KKI',
      'LEUVEN_1' : 'LEUVEN',
      'LEUVEN_2' : 'LEUVEN',
      'MAX_MUN' : 'MAX_MUN',
      'NYU' : 'NYU',
      'OLIN' : 'OLIN',
      'PITT' : 'PITT',
      'SBL' : 'SBL',
      'SDSU' : 'SDSU',
      'TRINITY' : 'TRINITY',
      'UCLA_1' : 'UCLA',
      'UCLA_2' : 'UCLA',
      'UM_1' : 'UM',
      'UM_2' : 'UM',
      'USM' : 'USM',
      'YALE' : 'YALE'

  }
  autis_df = pd.read_excel(f"data/ABIDE_1_Autism.xlsx")
  aspgr_df = pd.read_excel(f"data/ABIDE_2_Asperger.xlsx")
  adhd1_df = pd.read_excel(f"data/ADHD_1_ADHD1.xlsx")
  adhd2_df = pd.read_excel(f"data/ADHD_2_ADHD2.xlsx")
  adhd3_df = pd.read_excel(f"data/ADHD_3_ADHD3.xlsx")
  for i in range(autis_df.shape[0]):
    row = autis_df.iloc[i]
    sites.append(name[row['SITE_ID']])

  for i in range(aspgr_df.shape[0]):
    row = aspgr_df.iloc[i]
    sites.append(name[row['SITE_ID']])

  for i in range(adhd1_df.shape[0]):
    row = adhd1_df.iloc[i]
    sites.append(site_map[int(row['Site'])])

  for i in range(adhd2_df.shape[0]):
    row = adhd2_df.iloc[i]
    sites.append(site_map[int(row['Site'])])

  for i in range(adhd3_df.shape[0]):
    row = adhd3_df.iloc[i]
    sites.append(site_map[int(row['Site'])])

  return sites

def split_discovery_vs_replication(df):
  # 'OLIN': ASD 0
  # 'WU' : ADHD 0

  replication_sites = ['LEUVEN', 'MAX_MUN', 'TRINITY', 'SBL', 'NEURO']
  discovery_sites = ['CALTECH', 'CMU', 'NYU', 'KKI', 'PITT', 'SDSU', 'OLIN', 'UCLA', 'YALE', 'USM', 'UM', 'OREGON', 'PEKING', 'WU']
  # discovery_sites = ['CALTECH', 'CMU', 'KKI', 'LEUVEN', 'MAX-MUN', 'KKI', 'UM', 'TRINITY', 'NEURO', 'OREGON', 'SDSU']
  # replication_sites = ['NYU', 'PITT', 'SBL', 'UCLA', 'USM', 'YALE', 'PEKING']

  discovery_df = df[df['site'].isin(discovery_sites)]
  replication_df = df[df['site'].isin(replication_sites)]

  return discovery_df, replication_df

def store_dataframe(df, file_name):
  df.to_csv(file_name, index=False)

if __name__ == "__main__":
  
    dataset = BrainDataset(f"data/adhd_autism.npz")
    # map_df = pd.read_excel(f"data/mapping_from_190_to_200_ROIs.xlsx")
    # net_df = clean_net_df(pd.read_excel(f"{SOURCE_DIR}/brainnet/ABIDE_CC200_ROI_labels_netID.xlsx"), map_df)

    sites = extract_site_info()
    dataset.add_sites(sites)
    
    flattened_data = np.array([flatten_upper_triangle(data) for data in dataset.data])


    disorders_df = pd.DataFrame(flattened_data)
    disorders_df['label'] = dataset.label_name
    disorders_df['site'] = dataset.sites

    disorders_df = convert_to_asd_adhd(disorders_df)
    
    sites = extract_site_info()
    dataset.add_sites(sites)

    flattened_data = np.array([flatten_upper_triangle(data) for data in dataset.data])

    disorders_df = pd.DataFrame(flattened_data)
    disorders_df['label'] = dataset.label_name
    disorders_df['site'] = dataset.sites

    disorders_df = convert_to_asd_adhd(disorders_df)

    discovery_df, replication_df = split_discovery_vs_replication(disorders_df)
    print("discovery data: ", discovery_df.shape[0])
    print("replication data: ", replication_df.shape[0])
    
    discovery_df = discovery_df.drop(columns=['site'])
    replication_df = replication_df.drop(columns=['site'])

    store_dataframe(discovery_df, "data/discovery_df.csv")
    store_dataframe(replication_df, "data/replication_df.csv")