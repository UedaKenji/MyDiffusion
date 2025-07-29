import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionBlock(nn.Module):
    def __init__(self, nunits, batchnorm: bool = True, nunit2=None):
        super(DiffusionBlock, self).__init__()
        if nunit2 is not None:
            self.linear = nn.Linear(nunits, nunit2)
        else:
            self.linear = nn.Linear(nunits, nunits)
        # バッチ正規化用レイヤー
        if batchnorm:
            self.bn = nn.BatchNorm1d(nunits)
        else:
            self.bn = nn.Identity()
        # 活性化を一応モジュール化しておくと見やすい
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor):
        # x.shape = (batch_size, nunits)
        x = self.linear(x)
        x = self.bn(x)  # BatchNorm1d に通す
        x = self.activation(x)  # SiLU
        return x


class DiffusionModel(nn.Module):
    def __init__(
        self,
        nfeatures: int,
        ncondition=0,
        nblocks: int = 2,
        nunits: int = 32,
        finalunit=None,
        batch_norm: bool = False,
    ):
        super(DiffusionModel, self).__init__()

        self.nfeatures = nfeatures
        self.ncondition = ncondition
        # 入力層
        self.inblock = nn.Linear(nfeatures + 1 + ncondition, nunits)  # => (batch_size, nunits)
        # 中間ブロックを複数まとめる

        if finalunit is not None:
            self.midblocks = nn.ModuleList([DiffusionBlock(nunits, batch_norm) for _ in range(nblocks - 1)])
            self.midblocks.append(DiffusionBlock(nunits, batch_norm, finalunit))
            self.outblock = nn.Linear(finalunit, nfeatures)
        else:
            self.midblocks = nn.ModuleList([DiffusionBlock(nunits, batch_norm) for _ in range(nblocks)])
            self.outblock = nn.Linear(nunits, nfeatures)

        # 出力層

    def forward(self, x: torch.Tensor, t: torch.Tensor, c=None) -> torch.Tensor:
        """
        x: 入力データ
        t: ノイズレベル
        c: 条件
        """
        # x.shape = (batch_size, nfeatures)
        # t.shape = (batch_size, 1)
        if c is not None:
            val = torch.hstack([x, t, c])
        else:
            val = torch.hstack([x, t])  # => (batch_size, nfeatures+1)
        val = self.inblock(val)  # => (batch_size, nunits)

        for midblock in self.midblocks:
            val = midblock(val)  # => (batch_size, nunits)

        val = self.outblock(val)  # => (batch_size, nfeatures)
        return val


class ScoreModel:
    def __init__(
        self,
        nfeatures: int,
        nblocks: int = 2,
        nunits: int = 32,
        finalunit=None,
        batch_norm: bool = False,
    ):
        self.model = DiffusionModel(nfeatures, nblocks, nunits, finalunit, batch_norm)

    def predicted_noise(self, x, sigma_t):
        return self.model(x, sigma_t)

    def score_fn(self, x, sigma_t):
        return self.model(x, sigma_t) / sigma_t
