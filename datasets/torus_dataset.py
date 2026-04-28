import math

import pandas
import numpy as np
import torch
from torch.utils.data import Dataset

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.database.rcsb as rcsb
import biotite.structure.io.pdbx as pdbx

import torch.nn.functional as F

class BoardTorusDataset(Dataset):
    def __init__(self, N, seed=42, **kwargs):
        self.N = N
        self.seed = seed
        self.data = self.generate_all_samples(seed, N)
        self.data = self.wrap(self.data)
        self.data = self.data[:, None, :]

    @staticmethod
    def generate_all_samples(seed, N):
        generator = torch.Generator()
        generator.manual_seed(seed)

        x1 = torch.rand(N, generator=generator) * 4 - 2
        x2_ = (torch.rand(N, generator=generator) - torch.randint(high=2, size=(N,), generator=generator) * 2)
        x2 = x2_ + (torch.floor(x1) % 2)
        data = torch.cat([x1[:, None], x2[:, None]], dim=1)

        return data.float()

    def wrap(manifold, samples):
        return samples % (2 * torch.pi)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def dimension(self):
        return self.data.shape[-1]


class ProteinAngles(Dataset):
    def __init__(self, root, **kwargs):
        self.root = root
        self.data = torch.tensor(self.read_tsv(), dtype=torch.float32)
        self.data = self.wrap(self.data)
        self.data = self.data[:, None, :]

    def read_tsv(self):
        # 'source', 'phi', 'psi', 'amino'
        df = pandas.read_csv(self.root, sep='\t', header=None)
        col_1 = df[1] / 180 * math.pi + math.pi
        col_2 = df[2] / 180 * math.pi + math.pi
        return np.stack([col_1.to_numpy(), col_2.to_numpy()], axis=-1)

    def wrap(self, samples):
        return samples % (2 * torch.pi)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def dimension(self):
        return self.data.shape[-1]


class RNAAngles(Dataset):
    def __init__(self, root, **kwargs):
        self.root = root
        self.data = torch.tensor(self.read_tsv(), dtype=torch.float32)
        self.data = self.wrap(self.data)
        self.data = self.data[:, None, :]

    def read_tsv(self):
        # 'source', 'base', 'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'chi'
        df = pandas.read_csv(self.root, sep='\t', header=None)
        cols = []
        for i in range(2, len(df.columns)):
            cols.append(df[i] / 180 * math.pi + math.pi)
        return np.stack(cols, axis=-1)

    def wrap(self, samples):
        return samples % (2 * torch.pi)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def dimension(self):
        return self.data.shape[-1]

class SideChainAngleDataset(Dataset):
    def __init__(self, **kwargs):
        backbone_angles, side_chain_angles, sequences = self.read_pdbs()
        self.backbone_angles = backbone_angles
        self.side_chain_angles = side_chain_angles
        self.sequences = sequences
    
    def read_pdbs(self):
        file_name = "./pdb_s40_ids.txt"

        AA_MAP = {
            'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
            'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
            'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
            'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
        }
        backbone_arrays = []
        side_chain_arrays = []
        sequence_arrays = []
        with open(file_name) as f:
            for line in f.readlines():
                #download from pdb database
                file_path = rcsb.fetch(line[:4], format="pdb", target_path="./data")
                if line[5].isalpha():
                    file = pdb.PDBFile.read(file_path)
                    structure = file.get_structure(model=1)
                    chain = structure[structure.chain_id == line[5]]
                    
      
                    phi, psi, omega = struc.dihedral_backbone(chain) #get backbone dihedral angles

                    valid_mask = ~np.isnan(phi) & ~np.isnan(psi)

                    x_backbone = np.stack([phi[valid_mask], psi[valid_mask]], axis=-1)

                    backbone_arrays.append(x_backbone)


                    chi = struc.dihedral_side_chain(chain) #get chi angles
                    x_sidechain = np.nan_to_num(chi[valid_mask])

                    side_chain_arrays.append(x_sidechain)

                    ids, names = struc.get_residues(chain)

                    
                    x_sequence = names[valid_mask]

                    #one hot encode the amino acids

                    x_indices = np.array([AA_MAP.get(res_name, 20) for res_name in x_sequence])


                    x_seq_one_hot = F.one_hot(x_indices, num_classes=21).float()

                    sequence_arrays.append(x_seq_one_hot)
                    
                    

                else:
                    cif_file = pdbx.CIFFile.read(file_path)

                    # 2. Convert to AtomArray
                    atoms = pdbx.get_structure(cif_file, model=1)

                    # 3. Filter by Entity ID (Note: these are strings '1', '2', etc.)
                    entity_1 = atoms[atoms.entity_id == "1"]

                    phi, psi, omega = struc.dihedral_backbone(entity_1) #get backbone dihedral angles

                    valid_mask = ~np.isnan(phi) & ~np.isnan(psi)

                    x_backbone = np.stack([phi[valid_mask], psi[valid_mask]], axis=-1)
                    backbone_arrays.append(x_backbone)

                    chi = struc.dihedral_side_chain(entity_1) #get chi angles
                    x_sidechain = np.nan_to_num(chi[valid_mask])
                    side_chain_arrays.append(x_sidechain)

                    ids, names = struc.get_residues(entity_1)

                    
                    x_sequence = names[valid_mask]
                    sequence_arrays.append(x_sequence)
                    #one hot encode the amino acids

                    x_indices = np.array([AA_MAP.get(res_name, 20) for res_name in x_sequence])

        
                    x_seq_one_hot = F.one_hot(x_indices, num_classes=21).float()
        return backbone_arrays, side_chain_arrays, sequence_arrays
                

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.backbone_angles[idx], self.side_chain_angles[idx], self.sequences[idx]

if __name__ == "__main__":
    data = SideChainAngleDataset()


                
