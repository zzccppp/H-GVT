import io

import h5py
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import numpy as np

class LatentH5Dataset(Dataset):
    def __init__(self, h5_file_path):
        self.h5_file_path = h5_file_path

        with h5py.File(self.h5_file_path, "r") as f:
            self.length = len(f["embed_ind"])
            self.vae_config = torch.load(
                io.BytesIO(f["var_checkpoint/data"][:].tobytes()),
                weights_only=False,
                map_location="cpu",
            )["config"]

            self.all_multi_scale = f["embed_ind_multi_scale"][:].copy()
            self.all_num_nodes = f["num_nodes"][:].copy()

            embed_ind_dset = f["embed_ind"]
            self.all_embed_ind = [embed_ind_dset[i].copy() for i in range(self.length)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        multi_scale = self.all_multi_scale[idx]
        num_nodes = self.all_num_nodes[idx]
        embed_ind = self.all_embed_ind[idx]

        return {
            "multi_scale": torch.from_numpy(multi_scale).long(),
            "num_nodes": torch.tensor(num_nodes, dtype=torch.long),
            "embed_ind": torch.from_numpy(embed_ind).long(),
        }

    def get_var_ckpt(self):
        with h5py.File(self.h5_file_path, "r") as f:
            ckpt_bytes = f["var_checkpoint/data"][:].tobytes()

        buffer = io.BytesIO(ckpt_bytes)
        ckpt = torch.load(buffer, map_location="cpu", weights_only=False)
        return ckpt

    @staticmethod
    def collate_fn(list_of_data, padding_value=-1, eos_value=-1):
        batch_multi_scale = torch.stack(
            [data["multi_scale"] for data in list_of_data], dim=0
        )
        batch_num_nodes = torch.stack(
            [data["num_nodes"] for data in list_of_data], dim=0
        )
        embed_ind_list = [
            torch.cat([data["embed_ind"], torch.tensor([eos_value])])
            for data in list_of_data
        ]

        batch_embed_ind = pad_sequence(
            embed_ind_list, batch_first=True, padding_value=padding_value
        )

        return {
            "multi_scale": batch_multi_scale,
            "num_nodes": batch_num_nodes,
            "embed_ind": batch_embed_ind,
        }

    @staticmethod
    def collate_fn_normal_gpt(
        list_of_data, padding_value=-1, eos_value=-1, sos_value=0
    ):
        batch_num_nodes = torch.stack(
            [data["num_nodes"] for data in list_of_data], dim=0
        )
        embed_ind_list = [
            torch.cat(
                [
                    torch.tensor([sos_value]),
                    data["multi_scale"],
                    data["embed_ind"],
                    torch.tensor([eos_value]),
                ]
            )
            for data in list_of_data
        ]

        src = [data[:-1] for data in embed_ind_list]
        tgt = [data[1:] for data in embed_ind_list]

        src_tensors = pad_sequence(src, batch_first=True, padding_value=padding_value)
        tgt_tensors = pad_sequence(tgt, batch_first=True, padding_value=padding_value)

        return {
            "num_nodes": batch_num_nodes,
            "src": src_tensors,
            "tgt": tgt_tensors,
        }
