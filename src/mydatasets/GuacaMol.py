import os
import os.path as osp
from typing import Callable, List, Optional, Dict, Any, Set, Tuple

import torch
from tqdm import tqdm
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
)
from torch_geometric.utils import to_dense_adj, get_laplacian
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

try:
    from rdkit import Chem
    from rdkit.Chem import rdmolops
    from rdkit.Chem.rdchem import BondType as BT

    RDKitMol = Chem.rdchem.Mol
except ImportError:
    Chem = None
    BT = None
    RDKitMol = None
    print("RDKit not found. Please install RDKit to use this dataset.")
    print("You can install it via pip: pip install rdkit-pypi")
    print("Or via conda: conda install -c conda-forge rdkit")

class GuacaMolDataset(InMemoryDataset):

    TARGET_SPLITS = ["train", "valid", "test"]
    url_map = {
        "train": "https://ndownloader.figshare.com/files/13612760",
        "valid": "https://ndownloader.figshare.com/files/13612766",
        "test": "https://ndownloader.figshare.com/files/13612757",
    }

    BOND_TYPES_RDKIT = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE]
    ATOM_MAP = {
        "B": 0,
        "Br": 1,
        "C": 2,
        "Cl": 3,
        "F": 4,
        "I": 5,
        "N": 6,
        "O": 7,
        "P": 8,
        "S": 9,
        "Se": 10,
        "Si": 11,
    }

    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        if Chem is None or BT is None or RDKitMol is None:
            raise ImportError(
                "RDKit is required to use the GuacaMolDataset. Please install it."
            )

        self.current_split_name = split

        self.bond_map: Dict[Any, int] = {
            bond_type: i for i, bond_type in enumerate(self.BOND_TYPES_RDKIT)
        }
        self.num_bond_types: int = len(self.bond_map)

        self.atom_map: Dict[str, int] = self.ATOM_MAP.copy()
        self.num_atom_types: int = len(self.atom_map)

        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload
        )

        self.load(osp.join(self.processed_dir, self.processed_file_names[0]))

    def _get_processed_filename_for_split(self, split_name: str) -> str:
        return f"guacamol_{split_name}_processed_data_v1.pt"

    def _get_raw_filename_for_split(self, split_name: str) -> str:
        return f"guacamol_v1_{split_name}.smiles"

    @property
    def raw_file_names(self) -> List[str]:
        return [self._get_raw_filename_for_split(split) for split in self.TARGET_SPLITS]

    @property
    def processed_file_names(self) -> List[str]:
        return [
            self._get_processed_filename_for_split(self.current_split_name),
        ]

    def download(self) -> None:
        for split in self.TARGET_SPLITS:
            url = self.url_map[split]
            filename = self._get_raw_filename_for_split(split)
            download_url(url, self.raw_dir, filename=filename)
            print(f"Downloaded {filename} for split '{split}'.")

    def process(self) -> None:
        print("Processing raw SMILES data for all splits...")

        for split_name in self.TARGET_SPLITS:
            print(f"--- Processing split: {split_name} ---")
            raw_path = osp.join(
                self.raw_dir, self._get_raw_filename_for_split(split_name)
            )

            with open(raw_path, "r") as f:
                smiles_list = [line.strip() for line in f.readlines() if line.strip()]

            data_list = []
            pbar = tqdm(
                total=len(smiles_list), desc=f"Converting {split_name} SMILES to graphs"
            )

            for smiles_string in smiles_list:
                mol = Chem.MolFromSmiles(smiles_string)

                if mol is None:

                    pbar.update(1)
                    continue

                if mol.GetNumAtoms() < 2:
                    continue

                try:
                    Chem.SanitizeMol(mol)
                except Exception as e:

                    pbar.update(1)
                    continue

                Chem.Kekulize(mol)

                atom_features = []
                skip_mol = False
                for atom in mol.GetAtoms():
                    atom_symbol = atom.GetSymbol()
                    atom_idx = self.atom_map.get(atom_symbol)
                    if atom_idx is None:

                        skip_mol = True
                        break
                    atom_features.append(
                        torch.nn.functional.one_hot(
                            torch.tensor(atom_idx), num_classes=self.num_atom_types
                        )
                    )

                if skip_mol or not atom_features:
                    pbar.update(1)
                    continue

                x = torch.stack(atom_features, dim=0).float()

                edge_indices = []
                edge_attrs = []
                for bond in mol.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    bond_type_rdkit = bond.GetBondType()

                    bond_idx = self.bond_map.get(bond_type_rdkit)
                    if bond_idx is None:

                        continue

                    one_hot_bond = torch.nn.functional.one_hot(
                        torch.tensor(bond_idx), num_classes=self.num_bond_types
                    ).float()

                    edge_indices.extend([(i, j), (j, i)])
                    edge_attrs.extend([one_hot_bond, one_hot_bond])

                if edge_indices:
                    edge_index = (
                        torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                    )
                    edge_attr = torch.stack(edge_attrs, dim=0).float()
                else:

                    edge_index = torch.empty((2, 0), dtype=torch.long)
                    edge_attr = torch.empty((0, self.num_bond_types), dtype=torch.float)

                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    smiles=smiles_string,
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    pbar.update(1)
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()

            processed_path = osp.join(
                self.processed_dir, self._get_processed_filename_for_split(split_name)
            )
            self.save(data_list, processed_path)
            print(f"Processed data for split '{split_name}' saved to {processed_path}")

        print("All splits processed successfully.")

    def get_atom_type_map(self) -> Dict[str, int]:
        return self.atom_map

    def get_bond_type_map(self) -> Dict[Any, int]:
        return self.bond_map

    def get_num_atom_features(self) -> int:
        return self.num_atom_types

    def get_num_edge_features(self) -> int:
        return self.num_bond_types

def rcm_reorder_guacamol(data):
    adj_matrix = to_scipy_sparse_matrix(data.edge_index).tocsr()
    perm = reverse_cuthill_mckee(adj_matrix)[::-1]

    perm_tensor = torch.from_numpy(perm.copy()).long()
    inv_perm = torch.from_numpy(np.argsort(perm).copy()).long()

    data.x = data.x[perm_tensor]
    data.edge_index = inv_perm[data.edge_index]

    if "pe" in data.keys():
        data.pe = data.pe[perm_tensor]

    return data, perm

def guacamol_transform_pe_reorder(data, pe_dim=6):
    data, perm = rcm_reorder_guacamol(data)

    edge_index = data.edge_index
    num_nodes = data.num_nodes
    L = get_laplacian(edge_index, normalization="sym", num_nodes=num_nodes)
    L = to_dense_adj(L[0], edge_attr=L[1]).squeeze(0)

    try:
        eig_vals, eig_vecs = torch.linalg.eigh(L)
    except torch.linalg.LinAlgError:
        print(
            f"Warning: torch.linalg.eigh failed for graph with {num_nodes} nodes. Falling back to NumPy."
        )
        try:
            L_np = L.cpu().numpy()
            eig_vals_np, eig_vecs_np = np.linalg.eigh(L_np)
            eig_vals = torch.from_numpy(eig_vals_np).to(L.device).float()
            eig_vecs = torch.from_numpy(eig_vecs_np).to(L.device).float()
        except np.linalg.LinAlgError:

            print(
                f"Error: NumPy linalg.eigh also failed for graph with {num_nodes} nodes. Returning zero PE."
            )
            data.pe = torch.zeros((num_nodes, pe_dim), dtype=torch.float)
            return data

    k = min(pe_dim + 1, num_nodes)
    selected_eig_vecs = eig_vecs[:, 1:k]

    if selected_eig_vecs.shape[1] < pe_dim:
        padding_needed = pe_dim - selected_eig_vecs.shape[1]
        padding = torch.zeros(
            num_nodes, padding_needed, device=eig_vecs.device, dtype=torch.float
        )
        pe = torch.cat([selected_eig_vecs, padding], dim=1)
    else:
        pe = selected_eig_vecs

    assert pe.shape[1] == pe_dim, f"PE shape mismatch: {pe.shape[0]} != {num_nodes}"

    if pe.shape[1] > 0:

        mean = pe.mean(dim=0, keepdim=True)
        std = pe.std(dim=0, keepdim=True)

        pe = (pe - mean) / (std + 1e-6)

        pe = torch.nan_to_num(pe, nan=0.0)
    data.pe = pe.float()

    return data

def filter_fn(data: Data) -> bool:

    return data.edge_index.size(1) > 0

if __name__ == "__main__":
    root_dir = "data/GuacaMol"

    print("--- Loading Training Set ---")
    train_dataset = GuacaMolDataset(
        root=root_dir,
        split="train",
        force_reload=False,
        pre_transform=guacamol_transform_pe_reorder,
        pre_filter=None,
    )
    print(f"Number of graphs in training set: {len(train_dataset)}")
    print(f"Number of atom types: {train_dataset.get_num_atom_features()}")
    print(f"Number of bond types: {train_dataset.get_num_edge_features()}")
    print(f"Atom type map: {train_dataset.get_atom_type_map()}")
    print(f"Bond type map: {train_dataset.get_bond_type_map()}")
    if len(train_dataset) > 0:
        print("First graph in training set:")
        print(train_dataset[0])
        print(f"SMILES: {train_dataset[0].smiles}")

    print("\n--- Loading Validation Set ---")
    valid_dataset = GuacaMolDataset(root=root_dir, split="valid")
    print(f"Number of graphs in validation set: {len(valid_dataset)}")

    print("\n--- Loading Test Set ---")
    test_dataset = GuacaMolDataset(root=root_dir, split="test")
    print(f"Number of graphs in test set: {len(test_dataset)}")
