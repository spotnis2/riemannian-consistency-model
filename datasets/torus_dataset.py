import math

import pandas
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import os
from tqdm import tqdm

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
        self.pad_data_for_conditioning()
    
    def read_pdbs(self):
        files = ["./pdb_s40_ids.txt", "./bc40_ids.txt"]

        AA_MAP = {
            'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
            'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
            'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
            'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
        }
        backbone_arrays = []
        side_chain_arrays = []
        sequence_arrays = []
        structs_downloaded = 0
        for filename in tqdm(files, desc="Total Files", unit="file"):
            if structs_downloaded >= 1000:
                break
            with open(filename) as f:
                for line in tqdm(f, total=sum(1 for _ in open(filename))):
                    #download from pdb database
                    if structs_downloaded >= 1000:
                        break
                    if line[5].isalpha():
                        try:
                            file_path = rcsb.fetch(line[:4], format="pdb", target_path="./data")
                            file = pdb.PDBFile.read(file_path)
                            structure = file.get_structure(model=1)
                            
                            chain = structure[(structure.chain_id == line[5])]
                            
                            protein_mask = struc.filter_amino_acids(chain)
                            clean_chain = chain[protein_mask]
                            backbone = clean_chain[(np.isin(clean_chain.atom_name, ['N', 'C', 'CA', 'O']))].coord
                            

                        
                            _, x_sequence = struc.get_residues(clean_chain)

                            if len(backbone) != len(np.unique(clean_chain.res_id)) * 4:
                                # Log a warning or implement a filler/interpolation strategy
                                print(f"Warning: Missing backbone atoms")
                                continue
                            backbone_arrays.append(backbone)

                            sequence_arrays.append(x_sequence)

                        
                            chi = struc.dihedral_side_chain(chain) #get chi angles
                            
                            x_sidechain = np.nan_to_num(chi)

                            side_chain_arrays.append(x_sidechain)

                            os.remove(file_path)
                            structs_downloaded += 1
                        except Exception as e:
                            print(file_path)
                            print(f"Error encountered: {e}")

                    else:
                        try:
                            file_path = rcsb.fetch(line[:4], format="cif", target_path="./data")
                            cif_file = pdbx.CIFFile.read(file_path)
                        

                            atoms = pdbx.get_structure(cif_file, model=1)
                            
                            boolean_index = (atoms.chain_id == line[5])
                            # boolean_index = boolean_index[np.newaxis, ...]
                            chain = atoms[(boolean_index)]
                           
                            protein_mask = struc.filter_amino_acids(chain)
                            clean_chain = chain[protein_mask]
                            backbone = clean_chain[(np.isin(clean_chain.atom_name, ['N', 'C', 'CA', 'O']))].coord
                            if len(backbone) > 512:
                                backbone = backbone[:512]
                            
                            _, x_sequence = struc.get_residues(clean_chain)
                            if len(x_sequence) > 512:
                                x_sequence = x_sequence[:512]

                            if len(backbone) != len(np.unique(clean_chain.res_id)) * 4:
                                # Log a warning or implement a filler/interpolation strategy
                                print(f"Warning: Missing backbone atoms")
                                continue
                            backbone_arrays.append(backbone)

                            sequence_arrays.append(x_sequence)

                        
                            chi = struc.dihedral_side_chain(chain) #get chi angles
                            
                            x_sidechain = np.nan_to_num(chi)

                            if len(x_sidechain) > 512:
                                x_sidechain = x_sidechain[:512]

                            side_chain_arrays.append(x_sidechain)
                            os.remove(file_path) #space conservation
                            structs_downloaded += 1
                        except Exception as e:
                            print(file_path)
                            print(f"Error encountered: {e}")
                    
        return backbone_arrays, side_chain_arrays, sequence_arrays
                
    def pad_data_for_conditioning(self):
        # have to combine backbone arrays and sequence arrays to make conditioning vectors
        # will be using ProteinMPNN encoder
        # why? intuitively understands the 3D space of backbones, and using it to encode means we'll avoid steric clashes.
        max_len = max(len(arr) for arr in self.backbone_angles)
        batch_size = len(self.backbone_angles)

        batched_angles = np.zeros((batch_size, max_len, 3), dtype=np.float32)

        padding_mask = np.zeros((batch_size, max_len), dtype=np.float32)

        for i, arr in enumerate(self.backbone_angles):
            L = len(arr)

            batched_angles[i, :L, :] = arr

            padding_mask[i, :L] = 1.0

        tensor_angles = torch.tensor(batched_angles)
        tensor_mask = torch.tensor(padding_mask)

        #pad sequences
        max_len_sequences = max(len(arr) for arr in self.sequences)
        batch_size = len(self.sequences)

        batched_sequences = np.empty((batch_size, max_len_sequences), dtype=object)

        padding_mask_sequences = np.zeros((batch_size, max_len_sequences), dtype=np.float32)

        for i, arr in enumerate(self.sequences):
            L = len(arr)

            batched_sequences[i, :L] = arr

            padding_mask_sequences[i, :L] = 1.0       

       #pad sidechain 
        max_len_sc = max(len(arr) for arr in self.side_chain_angles)
        batch_size = len(self.side_chain_angles)

        batched_sc = np.empty((batch_size, max_len_sc, 4), dtype=np.float32)

        padding_mask_sc = np.zeros((batch_size, max_len_sc), dtype=np.float32)

        for i, arr in enumerate(self.side_chain_angles):
            L = len(arr)

            batched_sc[i, :L, :] = arr

            padding_mask_sc[i, :L] = 1.0  
        torch.save({'t1': tensor_angles, 't2': tensor_mask}, 'backbone_tensors.pt')
        np.savez('sequence_data.npz', names=batched_sequences, mask=padding_mask_sequences)
        np.savez('side_chain_data.npz', side_chains=batched_sc, mask=padding_mask_sc)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.backbone_angles[idx], self.side_chain_angles[idx], self.sequences[idx]

if __name__ == "__main__":
    data = SideChainAngleDataset()
        
class SideChainAngles(Dataset):
    def __init__(self, sc_path, cond_path, **kwargs):
        sc     = np.load(sc_path)
        angles = torch.tensor(sc['side_chains'], dtype=torch.float32)  # (N, L, 4)
        mask   = torch.tensor(sc['mask'],        dtype=torch.float32)  # (N, L)
        cond   = torch.load(cond_path, map_location='cpu')             # (N, L, 128)

        # align seq len bw angles and conditioning
        L = min(angles.shape[1], cond.shape[1])
        angles, mask, cond = angles[:, :L, :], mask[:, :L], cond[:, :L, :]

        # flatten to valid residues only
        valid       = mask.bool()           # (N, L)
        self.angles = angles[valid]         # (M, 4)
        self.cond   = cond[valid]           # (M, 128)

        # wrap to [0, 2π), residue dim --> (M, 1, 4) for RCM convention
        self.angles = (self.angles % (2 * torch.pi))[:, None, :]
        print(f"SideChainAngles: {len(self.angles)} valid residues")

    def __len__(self):
        return len(self.angles)

    def __getitem__(self, idx):
        return self.angles[idx], self.cond[idx]  # (1, 4), (128,)

    @property
    def dimension(self):
        return self.angles.shape[-1]  # 4