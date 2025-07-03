import scipy
import pandas as pd
import numpy as np
import os




def align_roi_to_190(data, roi_map):
  n = data.shape[2]
  aligned_data = np.full((190, 190, n),np.nan)

  mapper = {}
  for i in range(190):
    mapper[roi_map.iloc[i, 0]] = roi_map.iloc[i, 1]

  for i in range(190):
    for j in range(190):
      for k in range(n):
        if (i+1) in mapper and (j+1) in mapper:
          map_i = mapper[i+1]
          map_j = mapper[j+1]
          aligned_data[i][j][k] = data[map_i-1][map_j-1][k]
  return aligned_data

def balance_matrices(matrices):
    for corr_matrix in matrices:
        n = corr_matrix.shape[0]
        # Set diagonal to 1
        np.fill_diagonal(corr_matrix, 1)
        # Make symmetric: copy upper triangle to lower or vice versa
        for i in range(n):
            for j in range(i + 1, n):
                a, b = corr_matrix[i, j], corr_matrix[j, i]
                if a == 0 and b != 0:
                    corr_matrix[i, j] = b
                elif b == 0 and a != 0:
                    corr_matrix[j, i] = a
    return matrices


def extract_ids(data, disease):
  if(disease == "Alzheimer"):
    return list(data['Subject ID'].values)
  elif(disease == "Early-MCI"):
    return list(data['Subject ID'].values)
  elif(disease == "Late-MCI"):
    return list(data['Subject ID'].values)
  elif(disease == "Autism"):
    return list(data['SUB_ID'].values)
  elif(disease == "Asperger"):
    return list(data['SUB_ID'].values)
  elif(disease == "ADHD-Combined"):
    return list(data['ScanDir ID'].values)
  elif(disease == "ADHD-Hyperactive"):
    return list(data['ScanDir ID'].values)
  elif(disease == "ADHD-Inattentive"):
    return list(data['ScanDir ID'].values)
  elif(disease == "PTSD"):
    return list(data[data['Diagnosis']=='PTSD']['Subject no.'])
  elif(disease == "PCS+PTSD"):
    return list(data[data['Diagnosis']=='PCS+PTSD']['Subject no.'])


def store_data_as_npz(file_path):
    if not os.path.exists(file_path):
        np.savez(file_path, corr_matrix=merged_data, label=data_labels, id=data_ids)
        print("File saved successfully.")
    else:
        print("File already exists. No new file created.")

if __name__ == "__main__":
    SOURCE_DIR = "./data"

    
    print("Load input data")
    auts_mat = scipy.io.loadmat(f'{SOURCE_DIR}/sfc_abide_dc.mat')
    adhd_mat = scipy.io.loadmat(f'{SOURCE_DIR}/sfc_adhd_dc.mat')
    roi_190_to_200 = pd.read_excel(f'{SOURCE_DIR}/mapping_from_190_to_200_ROIs.xlsx')

    
    auts_data = auts_mat['sfc_abide_dc_autism']
    aspg_data = auts_mat['sfc_abide_dc_asperger']
    adh1_data = adhd_mat['sfc_adhd_dc_adhd1']
    adh2_data = adhd_mat['sfc_adhd_dc_adhd2']
    adh3_data = adhd_mat['sfc_adhd_dc_adhd3']

    print("SFC matrices size before alignment...")
    print("auts:", auts_data.shape)
    print("aspg:", aspg_data.shape)
    print("adh1:", adh1_data.shape)
    print("adh2:", adh2_data.shape)
    print("adh3:", adh3_data.shape)


    auts_data = align_roi_to_190(auts_data, roi_190_to_200)
    aspg_data = align_roi_to_190(aspg_data, roi_190_to_200)

    # Load subjects
    auts_subj = pd.read_excel(f'{SOURCE_DIR}/ABIDE_1_Autism.xlsx')
    aspg_subj = pd.read_excel(f'{SOURCE_DIR}/ABIDE_2_Asperger.xlsx')
    adh1_subj = pd.read_excel(f'{SOURCE_DIR}/ADHD_1_ADHD1.xlsx')
    adh2_subj = pd.read_excel(f'{SOURCE_DIR}/ADHD_2_ADHD2.xlsx')
    adh3_subj = pd.read_excel(f'{SOURCE_DIR}/ADHD_3_ADHD3.xlsx')

    # Reshape each dataset to (c, 200, 200)

    auts_tp = np.transpose(auts_data, (2, 0, 1))
    aspg_tp = np.transpose(aspg_data, (2, 0, 1))
    adh1_tp = np.transpose(adh1_data, (2, 0, 1))
    adh2_tp = np.transpose(adh2_data, (2, 0, 1))
    adh3_tp = np.transpose(adh3_data, (2, 0, 1))

    print("SFC matrices size after alignment...")
    print("auts:", auts_data.shape)
    print("aspg:", aspg_data.shape)
    print("adh1:", adh1_data.shape)
    print("adh2:", adh2_data.shape)
    print("adh3:", adh3_data.shape)

    # Combine the datasets into one array of shape (N, 200, 200)
    merged_data = np.concatenate([auts_tp,
                                aspg_tp,
                                adh1_tp,
                                adh2_tp,
                                adh3_tp], axis=0)

    labels = ['Autism',
            'Asperger',
            'ADHD-Combined',
            'ADHD-Hyperactive',
            'ADHD-Inattentive']

    # Create corresponding labels for each dataset
    data_labels = ([labels[0]]   *   auts_tp.shape[0]  +
                [labels[1]]   *   aspg_tp.shape[0]  +
                [labels[2]]   *   adh1_tp.shape[0]  +
                [labels[3]]   *   adh2_tp.shape[0]  +
                [labels[4]]   *   adh3_tp.shape[0]
    )

    # Create corresponding ids for each dataset
    data_ids = (
                extract_ids(auts_subj, labels[0]) +
                extract_ids(aspg_subj, labels[1]) +
                extract_ids(adh1_subj, labels[2]) +
                extract_ids(adh2_subj, labels[3]) +
                extract_ids(adh3_subj, labels[4])

    )
    

    merged_data = balance_matrices(merged_data)

    corr_matrix_label_id = [(merged_data, label, data_id) for merged_data, label, data_id in zip(merged_data, data_labels, data_ids)]

    
    file_path = f'{SOURCE_DIR}/adhd_autism.npz'
    store_data_as_npz(file_path)


    print("Finished!")