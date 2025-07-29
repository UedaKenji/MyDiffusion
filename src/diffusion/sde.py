import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from . import likelihood, nn_model
from .dataset import Mydataset


class Main:
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        self.model_config = config["nn_model"]
        self.train_config = config["train"]
        self.nn_model = nn_model.DiffusionModel(**self.model_config)
        self.nn_model.to(self.device)
        self.loss_fn = nn.MSELoss()
        self.sampler_config = config["sampler"]

        try:
            self.train_config["current_epoch"]
        except KeyError:
            self.train_config["current_epoch"] = 0

        # Optimizer の準備
        opt_cfg = self.train_config.get("optimizer", {})
        optim_cls = AdamW if opt_cfg.get("type", "Adam") == "AdamW" else optim.Adam
        self.optimizer = optim_cls(
            self.nn_model.parameters(),
            lr=self.train_config["lr"],
            weight_decay=opt_cfg.get("weight_decay", 0.0),
        )

        # Scheduler の準備
        sched_cfg = self.train_config.get("scheduler", None)
        if sched_cfg and sched_cfg["type"] == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=sched_cfg.get("factor", 0.5),
                patience=sched_cfg.get("patience", 10),
                min_lr=sched_cfg.get("min_lr", 1e-6),
            )
        else:
            self.scheduler = None

    def train(
        self,
        dataset: Mydataset,
        save_path: str = None,
        save_each_name: bool = None,
        device=None,
    ):

        if self.train_config["current_epoch"] + 1 >= self.train_config["nepochs"]:
            print("Training is already finished")
            return

        if save_path is not None:
            self.train_config["save_path"] = save_path
        if save_each_name is not None:
            self.train_config["save_each_name"] = save_each_name

        if device is None:
            device = self.device
        self.nn_model.to(device)

        train_loader = dataset.get_loader(
            batch_size=self.train_config["batch_size"],
            train_split=self.train_config["train_split"],
        )

        self.config["dataset_property"] = {}
        self.config["dataset_property"]["mean"] = dataset.mean
        self.config["dataset_property"]["std"]  = dataset.std
        self.config["dataset_property"]["nsample"] = len(dataset)
        self.config["dataset_property"]["nfeature"] = dataset.nfeature

        overwrite = self.train_config["overwrite"]
        save_name = self.train_config["save_path"] + ".pt"

        save = True
        if os.path.exists(save_name) and not overwrite:
            print(f"File {save_name} already exists. Overwrite? (y/n)")
            time.sleep(0.5)  # wait for user to
            pass
            ans = input()
            if ans == "y":
                overwrite = True
                self.save_model(save_name)
                print(f"Model saved at {save_name}")
            else:
                print("Quit training?")
                ans = input()
                if ans == "y":
                    return
                else:
                    overwrite = True
                    print("Continue training and don't save the model")
                    save = False

        n_checkpoints = self.train_config["ncheckpoints"]

        sigma = Sigma(self.sampler_config["sigma_min"], self.sampler_config["sigma_max"])

        def noiser(Xbatch, t):
            z = torch.randn(size=Xbatch.shape).to(device)
            noised = Xbatch + sigma(t) * z
            return noised, z

        for epoch in range(self.train_config["current_epoch"], self.train_config["nepochs"]):
            epoch_loss = 0.0
            steps = 0
            self.nn_model.train()
            for Xbatch in train_loader:
                Xbatch = Xbatch.to(device)
                timesteps = torch.rand(len(Xbatch), 1, device=device)
                noised, eps = noiser(Xbatch, timesteps.to(device))
                sigma_t = sigma(timesteps)

                pred = self.nn_model(noised, sigma_t)
                loss = self.loss_fn(pred, eps)

                self.optimizer.zero_grad()
                loss.backward()
                # ← ここで勾配クリッピング
                clip_grad_norm_(
                    self.nn_model.parameters(),
                    max_norm=self.train_config.get("max_grad_norm", 1.0),
                )
                self.optimizer.step()

                epoch_loss += loss.item()
                steps += 1
                self.train_config["current_epoch"] = epoch

            avg_loss = epoch_loss / steps
            if epoch % self.train_config["print_each"] == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch} loss = {avg_loss:.6f}, lr = {current_lr:.2e}")

            # ReduceLROnPlateau ならここでステップ
            if self.scheduler is not None:
                self.scheduler.step(avg_loss)

            # … save_model 呼び出しなど …
            if (epoch + 1) % n_checkpoints == 0 and epoch > 0 and save:
                if self.train_config["save_each_name"]:
                    save_name = f"{self.train_config['save_path']}_{epoch+1}.pt"
                else:
                    save_name = f"{self.train_config['save_path']}.pt"

                self.save_model(save_name)
                print(f"Model saved at {save_name}")

        """
        for epoch in range(self.train_config["current_epoch"], self.train_config["nepochs"]):
            epoch_loss = steps = 0
            for batch in train_loader:
                Xbatch = batch
                timesteps = torch.rand(size=[len(Xbatch), 1])   
                noised, eps = noiser(Xbatch, timesteps)
                sigma_timesteps = sigma(timesteps)
                #predicted_noise = self.model(noised.to(device), sigma_timesteps.to(device)) 
                predicted_noise = self.model(noised.to(device), sigma_timesteps.to(device)) 
                loss = self.loss_fn(predicted_noise, eps.to(device)).to(device)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                steps += 1
                self.train_config["current_epoch"] = epoch  

            if epoch % self.train_config["print_each"] == 0:    
                print(f"Epoch {epoch} loss = {epoch_loss / steps}")

            if (epoch+1) % n_checkpoints == 0 and save:
                if self.train_config["save_each_name"]:
                    save_name = f"{self.train_config['save_path']}_{epoch}.pt"
                else:
                    save_name = f"{self.train_config['save_path']}.pt"
            
                self.save_model(save_name)
                print(f"Model saved at {save_name}")
        """

    def train_Condtional(
        self,
        dataset: Mydataset,
        save_path: str = None,
        save_each_name: bool = None,
        device=None,
    ):

        if self.train_config["current_epoch"] + 1 >= self.train_config["nepochs"]:
            print("Training is already finished")
            return

        if save_path is not None:
            self.train_config["save_path"] = save_path
        if save_each_name is not None:
            self.train_config["save_each_name"] = save_each_name

        if device is None:
            device = self.device
        self.nn_model.to(device)

        train_loader = dataset.get_loader(
            batch_size=self.train_config["batch_size"],
            train_split=self.train_config["train_split"],
        )

        self.config["dataset_property"] = {}
        self.config["dataset_property"]["mean"] = dataset.mean
        self.config["dataset_property"]["std"] = dataset.std
        self.config["dataset_property"]["nsample"] = len(dataset)
        self.config["dataset_property"]["nfeature"] = dataset.nfeature

        overwrite = self.train_config["overwrite"]
        save_name = self.train_config["save_path"] + ".pt"

        save = True
        if os.path.exists(save_name) and not overwrite:
            print(f"File {save_name} already exists. Overwrite? (y/n)")
            time.sleep(0.5)  # wait for user to
            pass
            ans = input()
            if ans == "y":
                overwrite = True
                self.save_model(save_name)
                print(f"Model saved at {save_name}")
            else:
                print("Quit training?")
                ans = input()
                if ans == "y":
                    return
                else:
                    overwrite = True
                    print("Continue training and don't save the model")
                    save = False

        n_checkpoints = self.train_config["ncheckpoints"]

        sigma = Sigma(self.sampler_config["sigma_min"], self.sampler_config["sigma_max"])

        def noiser(Xbatch, t):
            z = torch.randn(size=Xbatch.shape)
            noised = Xbatch + sigma(t) * z
            return noised, z

        h_batch_size = int(0.5 * self.train_config["batch_size"])
        nfeatures = self.config["nn_model"]["nfeatures"]

        for epoch in range(self.train_config["current_epoch"], self.train_config["nepochs"]):
            epoch_loss = steps = 0
            for batch in train_loader:
                Xbatch = batch[:, :nfeatures]
                Cbatch = batch[:, nfeatures:]
                timesteps = torch.rand(size=[len(Xbatch), 1])
                noised, eps = noiser(Xbatch, timesteps)
                sigma_timesteps = sigma(timesteps)
                # predicted_noise = self.model(noised.to(device), sigma_timesteps.to(device))
                predicted_noise = self.nn_model(noised.to(device), sigma_timesteps.to(device), Cbatch.to(device))
                loss = self.loss_fn(predicted_noise, eps.to(device)).to(device)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss
                steps += 1
                self.train_config["current_epoch"] = epoch

            if epoch % self.train_config["print_each"] == 0:
                print(f"Epoch {epoch} loss = {epoch_loss / steps}")

            if (epoch + 1) % n_checkpoints == 0 and epoch > 0 and save:
                if self.train_config["save_each_name"]:
                    save_name = f"{self.train_config['save_path']}_{epoch+1}.pt"
                else:
                    save_name = f"{self.train_config['save_path']}.pt"

                self.save_model(save_name)
                print(f"Model saved at {save_name}")

    def sampling(self, nsamples: int, nstep=50, sigma_min=None, sigma_max=None, device=None):
        if sigma_min is None:
            sigma_min = self.sampler_config["sigma_min"]
        if sigma_max is not None:
            sigma_max = self.sampler_config["sigma_max"]

        if device is None:
            device = self.device

        self.nn_model.to(device)
        self.nn_model.eval()
        sigma = Sigma(sigma_min=sigma_min, sigma_max=sigma_max)
        nfeatures = self.config["nn_model"]["nfeatures"]

        dt = 1.0 / nstep
        time_schedule = torch.linspace(1.0, 0.0, nstep, device=device)

        """Sampler following the Denoising Diffusion Probabilistic Models method by Ho et al (Algorithm 2)"""
        with torch.no_grad():
            x = sigma(1) * torch.randn(size=(nsamples, nfeatures)).to(device)
            xt = [x]
            for t in time_schedule[1:]:
                sigma_t = sigma(t)
                sigma_tp = sigma(t + dt)

                predicted_noise = self.nn_model(x, sigma_tp * torch.ones(nsamples, 1).to(device))

                # See DDPM paper between equations 11 and 12
                d_sigma_sq = sigma_tp**2 - sigma_t**2
                x = x - 1 * d_sigma_sq * predicted_noise / sigma_t

                if t >= 0:
                    x += 1 * torch.sqrt(d_sigma_sq) * torch.randn(size=(nsamples, nfeatures)).to(device)
                xt += [x]
            return x, xt

    def sampling2(self, nsamples: int, nstep=50, sigma_min=None, sigma_max=None, device=None):
        if sigma_min is None:
            sigma_min = self.sampler_config["sigma_min"]
        if sigma_max is not None:
            sigma_max = self.sampler_config["sigma_max"]

        if device is None:
            device = self.device

        self.nn_model.to(device)
        self.nn_model.eval()
        sigma = Sigma(sigma_min=sigma_min, sigma_max=sigma_max)
        nfeatures = self.config["nn_model"]["nfeatures"]

        dt = 1.0 / nstep
        time_schedule = torch.linspace(1.0, 0.0, nstep, device=device)

        """Sampler following the Denoising Diffusion Probabilistic Models method by Ho et al (Algorithm 2)"""
        with torch.no_grad():
            x = sigma(1) * torch.randn(size=(nsamples, nfeatures)).to(device)
            xt = [x]
            for t in time_schedule[1:]:
                sigma_t = sigma(t)
                sigma_tp = sigma(t + dt)

                y = torch.tensor([-1.0, -1]) * torch.ones(nsamples, 2)
                y = y.to(device)

                predicted_noise = self.nn_model(x, sigma_tp * torch.ones(nsamples, 1).to(device), y)

                # See DDPM paper between equations 11 and 12
                d_sigma_sq = sigma_tp**2 - sigma_t**2
                x = x - 1 * d_sigma_sq * predicted_noise / sigma_t

                if t >= 0:
                    x += 1 * torch.sqrt(d_sigma_sq) * torch.randn(size=(nsamples, nfeatures)).to(device)
                xt += [x]
            return x, xt

    def sampling_wGP(
        self,
        nsamples: int,
        nstep=50,
        sigma_min=None,
        sigma_max=None,
        device=None,
        Kernel=None,
    ):
        if sigma_min is None:
            sigma_min = self.sampler_config["sigma_min"]
        if sigma_max is not None:
            sigma_max = self.sampler_config["sigma_max"]

        if device is None:
            device = self.device

        self.nn_model.to(device)
        self.nn_model.eval()
        sigma = Sigma(sigma_min=sigma_min, sigma_max=sigma_max, device=device)
        nfeatures = self.config["nn_model"]["nfeatures"]

        dt = 1.0 / nstep
        time_schedule = torch.linspace(1.0, 0.0, nstep, device=device)

        """Sampler following the Denoising Diffusion Probabilistic Models method by Ho et al (Algorithm 2)"""
        with torch.no_grad():
            x = sigma(1) * torch.randn(size=(nsamples, nfeatures)).to(device)
            xt = [x]
            for t in time_schedule[1:]:
                sigma_t = sigma(t)
                sigma_tp = sigma(t + dt)

                predicted_noise = self.nn_model(x, sigma_tp * torch.ones(nsamples, 1).to(device))

                # See DDPM paper between equations 11 and 12
                d_sigma_sq = sigma_tp**2 - sigma_t**2
                score = torch.zeros(nsamples, nfeatures).to(device)
                score[:] = predicted_noise / sigma_t

                score[:, 0] += 1 * torch.linalg.solve(Kernel + 1 * sigma_tp**2 * torch.eye(nsamples).to(device), x[:, 0])
                score[:, 1] += 1 * torch.linalg.solve(Kernel + 1 * sigma_tp**2 * torch.eye(nsamples).to(device), x[:, 1])

                x = x - 1 * d_sigma_sq * score
                if t >= 0:
                    x += 1 * torch.sqrt(d_sigma_sq) * torch.randn(size=(nsamples, nfeatures)).to(device)
                xt += [x]
            return x, xt

    def save_model(self, path, include_optimizer=True):
        ckpt = {
            "device": self.device,
            "nn_model": self.model_config,
            "model_state_dict": self.nn_model.state_dict(),
            "train_config": self.train_config,
            "sampler_config": self.sampler_config,
            "dataset_property": self.config["dataset_property"],
        }
        if include_optimizer:
            ckpt["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(ckpt, path)

    # checkpoint からモデルを読み込んでMainクラスのインスタンスを作成する
    @classmethod
    def load_model__old(cls, path):
        check_point = torch.load(Path(path), weights_only=True)
        config = {}
        config["device"] = check_point["device"]
        config["nn_model"] = check_point["nn_model"]
        config["train"] = check_point["train_config"]
        config["sampler"] = check_point["sampler_config"]

        main = cls(config)
        main.nn_model.load_state_dict(check_point["model_state_dict"])
        if "optimizer_state_dict" in check_point:
            main.optimizer = optim.Adam(main.nn_model.parameters(), lr=main.train_config["lr"])
        else:
            print("Note: optimizer state is not loaded")

        main.optimizer.load_state_dict(check_point["optimizer_state_dict"])
        main.config["dataset_property"] = check_point["dataset_property"]
        print(f"Model successfully loaded from {path}")
        return main

    @classmethod
    def load_model(cls, path):
        ckpt = torch.load(path, map_location="cpu")

        main = cls(
            {
                "device": ckpt["device"],
                "nn_model": ckpt["nn_model"],
                "train": ckpt["train_config"],
                "sampler": ckpt["sampler_config"],
            }
        )

        main.nn_model.load_state_dict(ckpt["model_state_dict"])
        main.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        main.config["dataset_property"] = ckpt["dataset_property"]
        if "scheduler_state_dict" in ckpt and main.scheduler is not None:
            main.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        if "optimizer_state_dict" in ckpt:
            main.optimizer = optim.Adam(main.nn_model.parameters(), lr=main.train_config["lr"])
        else:
            print("Note: optimizer state is not loaded")

        return main


class Main_conditional:
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        self.model_config = config["nn_model"]
        self.condition_config = config["condition"]
        if self.condition_config["enable_null"]:
            self.enable_null = True
            self.model_config["ncondition"] = self.condition_config["ninput"] + 1
        else:
            self.model_config["ncondition"] = self.condition_config["ninput"]
            self.enable_null = False
        self.train_config = config["train"]
        self.model = nn_model.DiffusionModel(**self.model_config)
        self.model.to(self.device)
        self.loss_fn = nn.MSELoss()
        self.sampler_config = config["sampler"]
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.train_config["lr"])

        try:
            self.train_config["current_epoch"]
        except KeyError:
            self.train_config["current_epoch"] = 0

    def train(
        self,
        dataset: Mydataset,
        save_path: str = None,
        save_each_name: bool = None,
        device=None,
    ):

        if self.train_config["current_epoch"] + 1 >= self.train_config["nepochs"]:
            print("Training is already finished")
            return

        if save_path is not None:
            self.train_config["save_path"] = save_path
        if save_each_name is not None:
            self.train_config["save_each_name"] = save_each_name

        if device is None:
            device = self.device
        self.model.to(device)

        train_loader = dataset.get_loader(
            batch_size=self.train_config["batch_size"],
            train_split=self.train_config["train_split"],
        )

        self.config["dataset_property"] = {}
        self.config["dataset_property"]["mean"] = dataset.mean
        self.config["dataset_property"]["std"] = dataset.std
        self.config["dataset_property"]["nsample"] = len(dataset)
        self.config["dataset_property"]["nfeature"] = dataset.nfeature

        overwrite = self.train_config["overwrite"]
        save_name = self.train_config["save_path"] + ".pt"

        save = True
        if os.path.exists(save_name) and not overwrite:
            print(f"File {save_name} already exists. Overwrite? (y/n)")
            time.sleep(0.5)  # wait for user to
            pass
            ans = input()
            if ans == "y":
                overwrite = True
                self.save_model(save_name)
                print(f"Model saved at {save_name}")
            else:
                print("Quit training?")
                ans = input()
                if ans == "y":
                    return
                else:
                    overwrite = True
                    print("Continue training and don't save the model")
                    save = False

        n_checkpoints = self.train_config["ncheckpoints"]

        sigma = Sigma(self.sampler_config["sigma_min"], self.sampler_config["sigma_max"])

        def noiser(Xbatch, t):
            z = torch.randn(size=Xbatch.shape)
            noised = Xbatch + sigma(t) * z
            return noised, z

        batch_size = self.train_config["batch_size"]
        p = 0.4
        h_batch_size = int(p * batch_size)
        nfeatures = self.config["nn_model"]["nfeatures"]

        for epoch in range(self.train_config["current_epoch"], self.train_config["nepochs"]):
            epoch_loss = steps = 0
            for batch in train_loader:
                batch_size = len(batch)
                Xbatch = batch[:, :nfeatures]
                Cbatch = batch[:, nfeatures : nfeatures + self.model_config["ncondition"]]

                if self.enable_null:
                    h_batch_size = int(p * batch_size)
                    null = torch.zeros(size=(batch_size, 1))
                    null[:h_batch_size] = 1
                    Cbatch[:h_batch_size] = 0
                    Cbatch = torch.hstack([Cbatch, null])

                timesteps = torch.rand(size=[batch_size, 1])
                noised, eps = noiser(Xbatch, timesteps)
                sigma_timesteps = sigma(timesteps)
                # predicted_noise = self.model(noised.to(device), sigma_timesteps.to(device))
                predicted_noise = self.model(noised.to(device), sigma_timesteps.to(device), Cbatch.to(device))
                loss = self.loss_fn(predicted_noise, eps.to(device)).to(device)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss
                steps += 1
                self.train_config["current_epoch"] = epoch

            if epoch % self.train_config["print_each"] == 0:
                print(f"Epoch {epoch} loss = {epoch_loss / steps}")

            if (epoch + 1) % n_checkpoints == 0 and epoch > 0 and save:
                if self.train_config["save_each_name"]:
                    save_name = f"{self.train_config['save_path']}_{epoch}.pt"
                else:
                    save_name = f"{self.train_config['save_path']}.pt"

                self.save_model(save_name)
                print(f"Model saved at {save_name}")

    def sampling_null(self, nsamples: int, nstep=50, sigma_min=None, sigma_max=None, device=None):
        if sigma_min is None:
            sigma_min = self.sampler_config["sigma_min"]
        if sigma_max is not None:
            sigma_max = self.sampler_config["sigma_max"]

        if device is None:
            device = self.device

        self.model.to(device)
        self.model.eval()
        sigma = Sigma(sigma_min=sigma_min, sigma_max=sigma_max)
        nfeatures = self.config["nn_model"]["nfeatures"]

        dt = 1.0 / nstep
        time_schedule = torch.linspace(1.0, 0.0, nstep, device=device)

        condition = torch.zeros(size=(nsamples, self.model_config["ncondition"])).to(device)
        condition[:, self.model_config["ncondition"] - 1] = 1
        print(condition.shape)

        """Sampler following the Denoising Diffusion Probabilistic Models method by Ho et al (Algorithm 2)"""
        with torch.no_grad():
            x = sigma(1) * torch.randn(size=(nsamples, nfeatures)).to(device)
            xt = [x]
            for t in time_schedule[1:]:
                sigma_t = sigma(t)
                sigma_tp = sigma(t + dt)

                predicted_noise = self.model(x, sigma_tp * torch.ones(nsamples, 1).to(device), condition)

                # See DDPM paper between equations 11 and 12
                d_sigma_sq = sigma_tp**2 - sigma_t**2
                x = x - 1 * d_sigma_sq * predicted_noise / sigma_t

                if t >= 0:
                    x += 1 * torch.sqrt(d_sigma_sq) * torch.randn(size=(nsamples, nfeatures)).to(device)
                xt += [x]
            return x, xt

    def sampling(
        self,
        nsamples: int,
        nstep=50,
        sigma_min=None,
        sigma_max=None,
        device=None,
        guiding_factor=1,
        condition=None,
    ):
        if sigma_min is None:
            sigma_min = self.sampler_config["sigma_min"]
        if sigma_max is not None:
            sigma_max = self.sampler_config["sigma_max"]

        if device is None:
            device = self.device

        self.model.to(device)
        self.model.eval()
        sigma = Sigma(sigma_min=sigma_min, sigma_max=sigma_max)
        nfeatures = self.config["nn_model"]["nfeatures"]

        dt = 1.0 / nstep
        time_schedule = torch.linspace(1.0, 0.0, nstep, device=device)

        ncondition = self.model_config["ncondition"]
        ninput = self.condition_config["ninput"]

        condition_tmp = torch.zeros(size=(nsamples, ncondition)).to(device)
        if len(condition.shape) == 1:
            # ncondition -> (nsample , ncondition+1)
            condition_tmp[:, :ninput] = condition
            condition = condition_tmp
            print(condition.shape)
        elif len(condition.shape) == 2:
            # (nsample , ncondition+1) -> (nsample , ncondition+1)
            condition_tmp[:, :ninput] = condition[:, :]
            condition = condition_tmp
            print(condition.shape)
        else:
            ValueError("condition shape is invalid")
        condition = condition.to(device)

        if type(guiding_factor) == torch.Tensor:
            guiding_factor = guiding_factor.to(device)
            if guiding_factor.dim() == 1:
                guiding_factor = guiding_factor.unsqueeze(1)

        if self.enable_null:
            null_condtion = torch.zeros(size=(nsamples, self.model_config["ncondition"])).to(device)
            null_condtion[:, ncondition - 1] = 1

        """Sampler following the Denoising Diffusion Probabilistic Models method by Ho et al (Algorithm 2)"""
        with torch.no_grad():
            x = sigma(1) * torch.randn(size=(nsamples, nfeatures)).to(device)
            xt = [x]
            for t in time_schedule[1:]:
                sigma_t = sigma(t)
                sigma_tp = sigma(t + dt)

                predicted_noise = self.model(x, sigma_tp * torch.ones(nsamples, 1).to(device), condition)

                if self.enable_null:

                    predicted_noise_null = self.model(x, sigma_tp * torch.ones(nsamples, 1).to(device), null_condtion)
                    predicted_noise = guiding_factor * predicted_noise + (1 - guiding_factor) * predicted_noise_null

                # See DDPM paper between equations 11 and 12
                d_sigma_sq = sigma_tp**2 - sigma_t**2
                x = x - 1 * d_sigma_sq * predicted_noise / sigma_t

                if t >= 0:
                    x += 1 * torch.sqrt(d_sigma_sq) * torch.randn(size=(nsamples, nfeatures)).to(device)
                xt += [x]
            return x, xt

    def sampling_wGP(
        self,
        nsamples: int,
        nstep=50,
        sigma_min=None,
        sigma_max=None,
        device=None,
        guiding_factor=1,
        condition=None,
        Kernel=None,
        mu=None,
    ):
        if sigma_min is None:
            sigma_min = self.sampler_config["sigma_min"]
        if sigma_max is not None:
            sigma_max = self.sampler_config["sigma_max"]

        if device is None:
            device = self.device

        self.model.to(device)
        self.model.eval()
        sigma = Sigma(sigma_min=sigma_min, sigma_max=sigma_max)
        nfeatures = self.config["nn_model"]["nfeatures"]

        dt = 1.0 / nstep
        time_schedule = torch.linspace(1.0, 0.0, nstep, device=device)

        ncondition = self.model_config["ncondition"]
        ninput = self.condition_config["ninput"]

        condition_tmp = torch.zeros(size=(nsamples, ncondition)).to(device)
        if len(condition.shape) == 1:
            # ncondition -> (nsample , ncondition+1)
            condition_tmp[:, :ninput] = condition
            condition = condition_tmp
            print(condition.shape)
        elif len(condition.shape) == 2:
            # (nsample , ncondition+1) -> (nsample , ncondition+1)
            condition_tmp[:, :ninput] = condition[:, :]
            condition = condition_tmp
            print(condition.shape)
        else:
            ValueError("condition shape is invalid")
        condition = condition.to(device)

        if type(guiding_factor) == torch.Tensor:
            guiding_factor = guiding_factor.to(device)
            if guiding_factor.dim() == 1:
                guiding_factor = guiding_factor.unsqueeze(1)

        if self.enable_null:
            null_condtion = torch.zeros(size=(nsamples, self.model_config["ncondition"])).to(device)
            null_condtion[:, ncondition - 1] = 1

        """Sampler following the Denoising Diffusion Probabilistic Models method by Ho et al (Algorithm 2)"""
        with torch.no_grad():
            x = sigma(1) * torch.randn(size=(nsamples, nfeatures)).to(device)
            xt = [x]
            for t in time_schedule[1:]:
                sigma_t = sigma(t)
                sigma_tp = sigma(t + dt)

                predicted_noise = self.model(x, sigma_tp * torch.ones(nsamples, 1).to(device), condition)

                if self.enable_null:

                    predicted_noise_null = self.model(x, sigma_tp * torch.ones(nsamples, 1).to(device), null_condtion)
                    predicted_noise = guiding_factor * predicted_noise + (1 - guiding_factor) * predicted_noise_null

                score = predicted_noise / sigma_t
                print(score.shape)
                score[:, 0] += 1 * torch.linalg.solve(
                    Kernel + 1 * sigma_tp**2 * torch.eye(nsamples).to(device),
                    x[:, 0] - mu,
                )
                score[:, 1] += 1 * torch.linalg.solve(
                    Kernel + 1 * sigma_tp**2 * torch.eye(nsamples).to(device),
                    x[:, 1] - mu,
                )

                d_sigma_sq = sigma_tp**2 - sigma_t**2
                x = x - 1 * d_sigma_sq * score

                if t >= 0:
                    x += 1 * torch.sqrt(d_sigma_sq) * torch.randn(size=(nsamples, nfeatures)).to(device)
                xt += [x]
            return x, xt

    def save_model(self, path, include_optimizer=True):
        check_point = {
            "device": self.device,
            "nn_model": self.model_config,
            "condition": self.condition_config,
            "model_state_dict": self.model.state_dict(),
            "train_config": self.train_config,
            "sampler_config": self.sampler_config,
            "dataset_property": self.config["dataset_property"],
        }
        if include_optimizer:
            check_point["optimizer_state_dict"] = self.optimizer.state_dict()
        torch.save(check_point, path)

    # checkpoint からモデルを読み込んでMainクラスのインスタンスを作成する
    @classmethod
    def load_model(cls, path):
        check_point = torch.load(Path(path), weights_only=True)
        config = {}
        config["device"] = check_point["device"]
        config["condition"] = check_point["condition"]
        config["nn_model"] = check_point["nn_model"]
        config["train"] = check_point["train_config"]
        config["sampler"] = check_point["sampler_config"]

        main = cls(config)
        main.model.load_state_dict(check_point["model_state_dict"])
        if "optimizer_state_dict" in check_point:
            main.optimizer = optim.Adam(main.model.parameters(), lr=main.train_config["lr"])
        else:
            print("Note: optimizer state is not loaded")

        main.optimizer.load_state_dict(check_point["optimizer_state_dict"])
        main.config["dataset_property"] = check_point["dataset_property"]
        print(f"Model successfully loaded from {path}")
        return main


class Sigma:
    def __init__(self, sigma_min, sigma_max, device=None):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_max_log = torch.log(torch.tensor(sigma_max))
        self.sigma_min_log = torch.log(torch.tensor(sigma_min))
        self.device = device

    def __call__(self, t):

        if type(t) == torch.Tensor:
            t = t.to(self.device)
        return self.sigma_min * torch.exp(t * (self.sigma_max_log - self.sigma_min_log))

    def __str__(self):
        return f"Sigma: {self.sigma_min} to {self.sigma_max}"


def conditional_sampling(
    model: nn_model.DiffusionModel,
    likelihood: likelihood.Likelihood,
    nsamples: int = 100,
    nstep=50,
    sigma_min=0.01,
    sigma_max=50,
    device=None,
    reject_outliers=False,
):

    sigma = Sigma(sigma_min=sigma_min, sigma_max=sigma_max, device=device)
    dt = 1.0 / nstep
    time_schedule = torch.linspace(1.0, 0.0, nstep, device=device)

    y_sigma_in = likelihood.y_sigma

    nfeatures = model.nfeatures

    """Sampler following the Denoising Diffusion Probabilistic Models method by Ho et al (Algorithm 2)"""
    x = sigma(1) * torch.randn(size=(nsamples, nfeatures)).to(device)
    xt = [x]

    # Asig = A @  torch.diag(sigma(0)*torch.ones(14))
    # Asig = A @  (sigma(0)*torch.ones(14)*dataset.std.squeeze())
    # y_sig20 = (Asig @ Asig.T)
    y_sig_est_sq0 = likelihood.y_estimated_sigma_sq(sigma(0))
    print(y_sig_est_sq0)
    print(y_sigma_in)
    model.to(device)

    for i, t in enumerate(time_schedule[1:]):
        x = x.requires_grad_(True)
        sigma_t = sigma(t)
        sigma_tp = sigma(t + dt)

        predicted_noise = model(x, sigma_tp * torch.ones(nsamples, 1).to(device))

        y_sig_est_sq = likelihood.y_estimated_sigma_sq(sigma_tp)

        x_hat = x - sigma_tp * predicted_noise

        x_grad = likelihood.x_grad(x, sigma_tp)

        d_sigma_sq = sigma_tp**2 - sigma_t**2
        x = x - 1 * d_sigma_sq * predicted_noise / sigma_t * 1

        grad_limit = 5 / sigma_tp
        x_grad[x_grad > grad_limit] = grad_limit
        x_grad[x_grad < -grad_limit] = -grad_limit

        if t < 1.0:
            # x = x - 1 * d_sigma_sq*x_grad/(y_sigma_in**2+0.*loss_mean)
            x = x - 1 * d_sigma_sq * x_grad * 1

        if reject_outliers and t >= 0:
            loss_sq = likelihood.loss_sq(x_hat, sigma_t)

            loss_mean = loss_sq.detach()
            loss_mean = loss_mean.mean(dim=0)

            loss = loss_sq.detach()
            # 外れ値は外れ値以外からランダムに選ぶ
            # outlier_index = 0.001*(loss/y_sigma_in**2).sum(dim=1) > y_sig2.sum()

            # print(y_sigma_in**2,y_sig2)

            outlier_index = 1 * (loss / y_sigma_in**2).sum(dim=1) > (1 + 10 * y_sig_est_sq / y_sigma_in**2).sum()
            # print(i,y_sig2.sum())
            if i % int(nstep / 50) == 0:
                # pass
                print(
                    i,
                    f"Outlier rate {100*outlier_index.sum()/nsamples:.2f}%",
                    t,
                    torch.sqrt(loss_mean),
                    torch.std(loss).detach(),
                    torch.max(loss).detach(),
                )

            x[outlier_index] = x[~outlier_index][torch.randint(0, x[~outlier_index].shape[0], (outlier_index.sum(),)), :]

        x += 1 * torch.sqrt(d_sigma_sq) * torch.randn(size=(nsamples, nfeatures)).to(device)

        x = x.detach_()
        # xt += [x_hat.detach()]
        xt += [x]
    return x, xt


def conditional_sampling2(
    model: nn_model.DiffusionModel,
    likelihood: likelihood.Likelihood,
    nsamples: int = 100,
    nstep=50,
    sigma_min=0.01,
    sigma_max=50,
    device=None,
    reject_outliers=False,
):

    sigma = Sigma(sigma_min=sigma_min, sigma_max=sigma_max, device=device)
    dt = 1.0 / nstep
    time_schedule = torch.linspace(1.0, 0.0, nstep, device=device)

    likelihood, likelihood2 = likelihood[0], likelihood[1]

    y_sigma_in = likelihood.y_sigma

    nfeatures = model.nfeatures

    """Sampler following the Denoising Diffusion Probabilistic Models method by Ho et al (Algorithm 2)"""
    x = sigma(1) * torch.randn(size=(nsamples, nfeatures)).to(device)
    xt = [x]

    # Asig = A @  torch.diag(sigma(0)*torch.ones(14))
    # Asig = A @  (sigma(0)*torch.ones(14)*dataset.std.squeeze())
    # y_sig20 = (Asig @ Asig.T)
    y_sig_est_sq0 = likelihood.y_estimated_sigma_sq(sigma(0))
    print(y_sig_est_sq0)
    print(y_sigma_in)
    model.to(device)

    for i, t in enumerate(time_schedule[1:]):
        x = x.requires_grad_(True)
        sigma_t = sigma(t)
        sigma_tp = sigma(t + dt)

        predicted_noise = model(x, sigma_tp * torch.ones(nsamples, 1).to(device))

        # Asig = A @ torch.diag(sigma_tp*torch.ones(14)*dataset.std.squeeze())
        # Asig = A @ (sigma_tp*torch.ones(14)*dataset.std.squeeze())

        y_sig_est_sq = likelihood.y_estimated_sigma_sq(sigma_tp)
        # y_sig2 = y_sig2.expand(nsamples,y_sig2.shape[0])

        x_hat = x - sigma_tp * predicted_noise

        loss_sq = likelihood.loss_sq(x_hat, sigma_t)

        # x_hat = dataset.inverse_normalize(x_hat)
        # loss = (measurement- (A@x_hat.T).T).pow(2)

        # n配列のy_sig2からnsaamples*n に変換
        # loss_mean = loss_sq.detach()
        # loss_mean = loss_mean.mean(dim=0)

        # log_likelihood = (0.5*loss / (0.9*y_sigma_in**2+0.1*loss_mean) ).sum(dim=1)
        # log_likelihood = (0.5*loss / (1*y_sigma_in**2+0.01*(y_sig2-y_sig20)) ).sum(dim=1)
        # log_likelihood = likelihood.log_likelihood(x_hat,sigma_t)

        x_grad = likelihood.x_grad(x, sigma_tp)

        x_grad += likelihood2.x_grad(x, sigma_tp)
        # x_grad = torch.autograd.grad(outputs=log_likelihood, inputs=x, grad_outputs=torch.ones_like(log_likelihood), create_graph=False)[0]
        # x = x.detach_()

        # See DDPM paper between equations 11 and 12
        d_sigma_sq = sigma_tp**2 - sigma_t**2
        x = x - 1 * d_sigma_sq * predicted_noise / sigma_t * 1

        grad_limit = 5 / sigma_tp
        x_grad[x_grad > grad_limit] = grad_limit
        x_grad[x_grad < -grad_limit] = -grad_limit

        if t < 1.0:
            # x = x - 1 * d_sigma_sq*x_grad/(y_sigma_in**2+0.*loss_mean)
            x = x - 1 * d_sigma_sq * x_grad * 1

        if reject_outliers and t >= 0:
            loss = loss_sq.detach()
            # 外れ値は外れ値以外からランダムに選ぶ
            # outlier_index = 0.001*(loss/y_sigma_in**2).sum(dim=1) > y_sig2.sum()

            # print(y_sigma_in**2,y_sig2)

            outlier_index = 1 * (loss / y_sigma_in**2).sum(dim=1) > (1 + 10 * y_sig_est_sq / y_sigma_in**2).sum()
            # print(i,y_sig2.sum())
            if i % int(nstep / 50) == 0:
                # pass
                loss_mean = loss_sq.detach()
                print(
                    i,
                    f"Outlier rate {100*outlier_index.sum()/nsamples:.2f}%",
                    t,
                    torch.sqrt(loss_mean),
                    torch.std(loss).detach(),
                    torch.max(loss).detach(),
                )

            x[outlier_index] = x[~outlier_index][torch.randint(0, x[~outlier_index].shape[0], (outlier_index.sum(),)), :]

        x += 1 * torch.sqrt(d_sigma_sq) * torch.randn(size=(nsamples, nfeatures)).to(device)

        x = x.detach_()
        # xt += [x_hat.detach()]
        xt += [x]

    return x, xt
