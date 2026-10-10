import hydra
import torch
from torch.utils.data import DataLoader
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import trange
from rdkit import Chem
import io
import h5py
from load_dataset import get_atom_types, get_dataset
from models.gpt2_modify import GPT_Normal
from models.gvar import GVAR
from mydatasets.LatentH5Dataset import LatentH5Dataset

from fcd_torch import FCD
import networkx as nx
from utils.chems import nspdk_stats
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.assess_distribution_learning import (
    assess_distribution_learning,
    _assess_distribution_learning,
)
import os
from moses import get_all_metrics

class PrecomputedMoleculeGenerator(DistributionMatchingGenerator):

    def __init__(self, molecule_list):
        self.molecules = molecule_list
        self.num_mols = len(molecule_list)
        self.index = 0

        if self.num_mols == 0:
            print("警告: 传入的分子列表为空！Guacamol 评估将会失败。")

    def generate(self, number_samples):
        samples = []

        for _ in range(number_samples):
            samples.append(self.molecules[self.index])
            self.index = (self.index + 1) % self.num_mols
        return samples

def check_validity(mol):
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True
    except:
        return False

def mols_to_nx(mols):
    nx_graphs = []
    for mol in mols:
        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), label=atom.GetSymbol())

        for bond in mol.GetBonds():
            G.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                label=int(bond.GetBondTypeAsDouble()),
            )

        nx_graphs.append(G)
    return nx_graphs

def tensor_to_mol(x, adj, atom_types, bond_types):

    x = x.cpu().numpy()
    adj = adj.cpu().numpy()

    mol = Chem.RWMol()

    for atom_type in x:
        atom = Chem.Atom(atom_types[atom_type])
        mol.AddAtom(atom)

    for i, j in zip(*adj.nonzero()):
        if i < j:
            bond_type = bond_types[adj[i, j]]
            mol.AddBond(int(i), int(j), bond_type)

    mol = mol.GetMol()

    return mol

def load_model(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ar_cfg = ckpt["config"]
    ar_model = GPT_Normal(ar_cfg.model.gpt)
    ar_model.load_state_dict(ckpt["model_state_dict"])
    ar_model.eval()

    h5_file_path = ar_cfg.dataset.save_path
    with h5py.File(h5_file_path, "r") as f:

        vae_config = torch.load(
            io.BytesIO(f["var_checkpoint/data"][:].tobytes()),
            weights_only=False,
            map_location="cpu",
        )["config"]

        ckpt_bytes = f["var_checkpoint/data"][:].tobytes()

    buffer = io.BytesIO(ckpt_bytes)
    vae_ckpt = torch.load(buffer, map_location="cpu", weights_only=False)

    vae_model = GVAR(vae_config.gvar)
    vae_model.load_state_dict(vae_ckpt["model_state_dict"])
    vae_model.eval()

    return ar_model, vae_model, vae_ckpt["config"]["dataset"]["name"]

def sample_molecules(
    ar_model,
    vae_model,
    atom_types: dict,
    bond_types: dict,
    cfg: DictConfig,
    device: torch.device,
):

    total_samples = cfg.total_samples
    batch_size = cfg.batch_size

    all_mols = []
    valid_mols = []
    valid_smiles = []

    num_batches = (total_samples + batch_size - 1) // batch_size

    print(
        f"开始生成 {total_samples} 个分子, 分为 {num_batches} 批 (每批 {batch_size} 个)..."
    )

    for _ in trange(num_batches, desc="Sampling Batches"):

        current_batch_size = min(batch_size, total_samples - len(all_mols))
        if current_batch_size <= 0:
            break

        list_tokens = ar_model.sample(
            batch_size=current_batch_size,
            device=device,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            do_sample=cfg.do_sample,
            top_k=cfg.top_k,
        )

        codebook_size = vae_model.vq.codebook.shape[0]
        list_tokens_tensor = []
        for tokens in list_tokens:
            t = torch.tensor(tokens, dtype=torch.long, device=device)
            t = t[t < codebook_size]
            if t.numel() > 0:
                list_tokens_tensor.append(t)

        if len(list_tokens_tensor) == 0:
            continue

        with torch.no_grad():
            quantized_node, adj_quantized, mask_node, batch = vae_model.decode(
                list_tokens_tensor
            )

        for i in range(len(list_tokens_tensor)):
            x = quantized_node[batch == i]

            num_nodes = x.shape[0]
            if num_nodes == 0:
                continue

            adj_full = adj_quantized[i]
            if num_nodes > adj_full.shape[0]:

                continue

            adj_slice = adj_full[:num_nodes, :num_nodes]

            adj = (adj_slice + 1) % 4

            mol = tensor_to_mol(x, adj, atom_types, bond_types)
            all_mols.append(mol)

            if mol is None:
                continue

            try:

                Chem.SanitizeMol(mol)
                smiles = Chem.MolToSmiles(mol)

                if Chem.MolFromSmiles(smiles) is not None:
                    valid_mols.append(mol)
                    valid_smiles.append(smiles)

            except Exception as e:

                pass

    if total_samples > 0:

        valid_rate = len(valid_mols) / len(all_mols) if len(all_mols) > 0 else 0
    else:
        valid_rate = 0.0

    all_mols_filtered = [m for m in all_mols if m is not None]

    return all_mols_filtered, valid_mols, valid_smiles, valid_rate

def get_testset_smiles(dataset_name: str):
    if dataset_name == "QM9":
        dataset = get_dataset(dataset_name, "test")
        smiles = [i.smiles for i in dataset]
        mols = [Chem.MolFromSmiles(s) for s in smiles]
    elif dataset_name == "ZINC":
        dataset = get_dataset(dataset_name, "test")
        smiles = [i.smiles for i in dataset]
        mols = [Chem.MolFromSmiles(s) for s in smiles]
    else:
        raise NotImplementedError(
            f"Dataset {dataset_name} is not supported for test set retrieval."
        )

    return smiles, mols

@hydra.main(version_base=None, config_path="config", config_name="sample_ar")
def main(cfg: DictConfig) -> None:

    print("loading model from checkpoint:", cfg.ar_ckpt)
    ar_model, vae_model, dataset_name = load_model(cfg.ar_ckpt)
    print("Dataset Name:", dataset_name)

    device = torch.device(cfg.device)
    ar_model.to(device)
    vae_model.to(device)

    atom_types = get_atom_types(dataset_name)

    bond_types = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }

    import time

    start_time = time.time()

    all_mols, valid_mols, valid_smiles, valid_rate = sample_molecules(
        ar_model, vae_model, atom_types, bond_types, cfg, device
    )

    end_time = time.time()

    print(f"Time for generate: {end_time - start_time:.2f} secs")

    unique_smiles = list(set(valid_smiles))
    unique_rate = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0

    print("Dataset Name:", dataset_name)
    if dataset_name == "QM9" or dataset_name == "ZINC":
        test_smiles, test_mols = get_testset_smiles(dataset_name)

        fcd = FCD(device=cfg.device, n_jobs=16)
        fcd_score = fcd(gen=unique_smiles, ref=test_smiles)
        nspdk = nspdk_stats(mols_to_nx(test_mols), mols_to_nx(valid_mols))

        print(f"总分子数: {len(all_mols)}")
        print(f"有效分子数: {len(valid_mols)}")
        print(f"有效率: {valid_rate * 100:.2f}%")
        print(f"Unique rate: {unique_rate * 100:.2f}%")
        print(f"FCD Score: {fcd_score:.4f}")
        print(f"NSPDK MMD: {nspdk:.4f}")
    elif dataset_name == "MOSES":
        metrics = get_all_metrics(unique_smiles)
        print("Valid ", valid_rate)
        print("Unique ", unique_rate)
        print(
            "Novel ",
            metrics["Novelty"] * 100,
        )
        print(
            "Filters ",
            metrics["Filters"] * 100,
        )
        print("FCD ", metrics["FCD/Test"])
        print("SNN ", metrics["SNN/Test"])
        print("Scaf ", metrics["Scaf/Test"] * 10)
    elif dataset_name == "Guacamol":
        if not unique_smiles:
            raise ValueError(
                "no valid smiles generated, cannot perform Guacamol evaluation."
            )
        print("Valid ", valid_rate)
        print("Unique ", unique_rate)
        generator = PrecomputedMoleculeGenerator(unique_smiles)

        guacamol_train_path = "data/GuacaMol/raw/guacamol_v1_train.smiles"
        if hasattr(cfg, "train_path"):
            guacamol_train_path = cfg.train_path

        if not os.path.exists(guacamol_train_path):

            print(
                f"警告: 找不到 Guacamol 训练集文件: {guacamol_train_path}，请检查路径。"
            )

        output_dir = "guacamol_results.json"

        _assess_distribution_learning(
            generator,
            chembl_training_file=guacamol_train_path,
            json_output_file=output_dir,
            benchmark_version="v1",
            number_samples=len(unique_smiles),
        )

if __name__ == "__main__":
    main()
