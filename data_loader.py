import numpy as np
from torch.utils.data import Dataset

class BrainDataset(Dataset):
    def __init__(self, data_file):
        loaded      = np.load(data_file)
        self.data   = loaded['corr_matrix']      # Shape: (N, 200, 200)
        self.labels = loaded['label']    # Shape: (N,)
        self.ids    = loaded['id']        # Shape: (N,)
        self.sites = []

        # Convert string labels to numeric
        self.label_mapping = {
          'Autism'            : 0,
          'Asperger'          : 1,
          'ADHD-Combined'     : 2,
          'ADHD-Hyperactive'  : 3,
          'ADHD-Inattentive'  : 4,
        }
        self.label_name = self.labels
        self.labels = np.array([self.label_mapping[label] for label in self.labels])


    def save_data(self, file_path):
      if not os.path.exists(file_path):
          self.reverse_mapping = self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
          label_name = np.array([self.reverse_mapping[label] for label in self.labels])
          np.savez(file_path, corr_matrix=self.data, label=label_name, id=self.ids)
          print("File saved successfully.")
      else:
          print("File already exists. No new file created.")

    def add_sites(self, sites):
        self.sites = sites

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return each data slice as a tensor
        corr_matrix = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)
        label       = torch.tensor(self.labels[idx], dtype=torch.long)
        subject_id  = self.ids[idx]

        return corr_matrix, label, subject_id