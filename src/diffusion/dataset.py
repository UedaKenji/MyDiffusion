import json
from abc import ABC, abstractmethod

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class Mydataset(ABC):
    @abstractmethod
    def __init__(self, nfeature: int, ext_normalization=None, print_normalization=True):
        self.nfeature = nfeature
        if ext_normalization is not None:
            params = ext_normalization
        else:
            params = self.get_normalization_params()

        if isinstance(params, tuple):
            self.mean, self.std = params
        elif isinstance(params, dict):
            self.mean = params["mean"]
            self.std = params["std"]
        else:
            raise ValueError("Normalization parameters must be a tuple or a dictionary with 'mean' and 'std' keys.")

        if print_normalization:
            # self.name_list が存在する場合は、特徴量名も表示する.特徴名の列数はそろえること
            if "name_list" in self.__dict__:
                str_max = max([len(name) for name in self.name_list])
                for i in range(nfeature):
                    print(f"feature {i} {self.name_list[i].rjust(str_max)} mean:{self.mean[i]} std:{self.std[i]}")
            else:
                for i in range(nfeature):
                    print(f"feature {i} mean:{self.mean[i]} std:{self.std[i]}")

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def get_normalization_params(self):
        # データの正規化に必要なパラメータを返す
        # 例えば、平均値と標準偏差を返す場合
        # return torch.tensor([0.0]*self.nfeature), torch.tensdataset_propertyor([1.0]*self.nfeature)
        pass

    def get_all_data(self):
        # 全データを返す
        # return self.data   for example
        pass

    def normalize(self, data):
        return (data - self.mean) / self.std

    def inverse_normalize(self, data):
        if isinstance(data, list):
            return [d.to("cpu") * self.std + self.mean for d in data]
        elif isinstance(data, torch.Tensor):
            data = data.to("cpu")
            return data * self.std + self.mean

    def denormalize(self, data):
        return self.inverse_normalize(data)

    def get_loader(self, data=None, batch_size=128, train_split=1.0, shuffle=True):

        self.batch_size = batch_size

        if data is None:
            if "data" not in self.__dict__:
                raise ValueError("data is None")
            data = self.data

        full_length = self.__len__()
        train_length = int(full_length * train_split)
        val_length = full_length - train_length

        train_dataset, val_dataset = random_split(self.normalize(data), [train_length, val_length])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        if train_split < 1.0:
            return train_loader, val_loader
        else:
            return train_loader


class Mydataset_conditional(ABC):
    @abstractmethod
    def __init__(self, nfeature: int, ext_normalization=None, print_normalization=True):
        self.nfeature = nfeature
        if ext_normalization is not None:
            self.mean, self.std = ext_normalization
        else:
            self.mean, self.std = self.get_normalization_params()

        if print_normalization:
            # self.name_list が存在する場合は、特徴量名も表示する.特徴名の列数はそろえること
            if "name_list" in self.__dict__:
                str_max = max([len(name) for name in self.name_list])
                for i in range(nfeature):
                    print(f"feature {i} {self.name_list[i].rjust(str_max)} mean:{self.mean[i]} std:{self.std[i]}")
            else:
                for i in range(nfeature):
                    print(f"feature {i} mean:{self.mean[i]} std:{self.std[i]}")

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def get_normalization_params(self):
        # データの正規化に必要なパラメータを返す
        # 例えば、平均値と標準偏差を返す場合
        # return torch.tensor([0.0]*self.nfeature), torch.tensdataset_propertyor([1.0]*self.nfeature)
        pass

    def get_all_data(self):
        # 全データを返す
        # return self.data   for example
        pass

    def normalize(self, data):
        return (data - self.mean) / self.std

    def inverse_normalize(self, data):
        data = data.to("cpu")
        return data * self.std + self.mean

    def denormalize(self, data):
        return self.inverse_normalize(data)

    def get_loader(self, data=None, batch_size=128, train_split=1.0, shuffle=True):

        self.batch_size = batch_size

        if data is None:
            if "data" not in self.__dict__:
                raise ValueError("data is None")
            data = self.data

        full_length = self.__len__()
        train_length = int(full_length * train_split)
        val_length = full_length - train_length

        all_dataset = torch.concatenate([self.normalize(data), self.cond_data], dim=1)

        train_dataset, val_dataset = random_split(all_dataset, [train_length, val_length])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        if train_split < 1.0:
            return train_loader, val_loader
        else:
            return train_loader


def load_metadata(filepath, key):
    """ファイルから指定されたメタデータを読み込む"""
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith(f"# meta:{key}="):
                json_str = line.strip().split("=", 1)[1]
                return json.loads(json_str)
    raise ValueError(f"Metadata key '{key}' not found in {filepath}")


def check_metadata(filepath, current_metadata):
    """既存ファイルとメタデータが一致するかチェック"""
    for key, val in current_metadata.items():
        existing = load_metadata(filepath, key)
        if existing != val:
            raise ValueError(f"Metadata mismatch for '{key}': {existing} != {val}")


def write_metadata_and_header(filepath, metadata_dict, columns):
    """メタデータとカラムを含むヘッダーを書き込む"""
    with open(filepath, "w") as f:
        for key, val in metadata_dict.items():
            f.write(f"# meta:{key}={json.dumps(val)}\n")
        f.write(",".join(columns) + "\n")


def append_data(filepath, data_rows):
    """データをCSVに追記"""
    df = pd.DataFrame(data_rows)
    df.to_csv(filepath, mode="a", header=False, index=False)
