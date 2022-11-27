from __future__ import annotations
from rich.progress import track
from console import console
import datetime
import os
import random
import shutil
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as nn
import config as cfg
from timeit import default_timer as timer
from torch.utils.tensorboard import SummaryWriter

plt.style.use("fivethirtyeight")

console.log(os.path.abspath(os.path.dirname(__file__)))


class Engine:
    def __init__(self, model, loss_fn, optimizer, ckpt_interval):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.ckpt_interval = ckpt_interval
        self.device = cfg.DEVICE
        self.model.to(self.device)
        self.train_loader = None
        self.val_loader = None
        self.writer = None
        self.is_load_checkpoint = False
        self.losses = []
        self.val_losses = []
        self.total_epochs = 0
        self.train_step = self._make_train_step_fn()
        self.val_step = self._make_val_step_fn()
        self.parent_folder_path = os.path.abspath(os.path.dirname(__file__))
        self.visualization = {}
        self.handles = {}

    def to(self, device):
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            console.log(
                f"Couldn't send it to {device}, \
                    sending it to {self.device} instead.",
            )
            self.model.to(self.device)

    def set_loaders(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_tensorboard(self, name, folder="runs"):
        suffix = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.writer = SummaryWriter(f"{folder}/{name}_{suffix}")

    def _make_train_step_fn(self):
        def perform_train_step_fn(x, y):

            self.model.train()

            yhat = self.model(x)

            loss = self.loss_fn(yhat, y)

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            return loss.item()

        return perform_train_step_fn

    def _make_val_step_fn(self):
        def perform_val_step_fn(x, y):
            self.model.eval()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            return loss.item()

        return perform_val_step_fn

    def _mini_batch(self, validation=False):
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step
        else:
            data_loader = self.train_loader
            step_fn = self.train_step

        if data_loader is None:
            return None

        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

        loss = np.mean(mini_batch_losses)

        return loss

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        try:
            self.train_loader.sampler.generator.manual_seed(seed)
        except AttributeError:
            pass

    def train(self, n_epochs, seed=42):
        os.makedirs(
            os.path.join(
                self.parent_folder_path,
                "checkpoints",
            ),
            exist_ok=True,
        )
        self.set_seed(seed)
        if not self.is_load_checkpoint:
            try:
                shutil.rmtree(
                    os.path.join(
                        self.parent_folder_path,
                        "checkpoints",
                    ),
                )
            except Exception as e:
                console.log(e)
                pass
        start_time = timer()
        # for epoch in tqdm(range(self.total_epochs, n_epochs)):
        for epoch in track(
            range(self.total_epochs, n_epochs), description="Iterating over dataset..."
        ):
            self.total_epochs += 1
            loss = self._mini_batch(validation=False)
            self.losses.append(loss)
            with torch.no_grad():
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)
            if epoch % self.ckpt_interval == 0:
                self.save_checkpoint(epoch)
                console.log(
                    f"[bold][cyan][INFO][/][/] Epoch: {epoch+1} | "
                    f"train_loss: {loss:.4f} | "
                    f"test_loss: {val_loss:.4f} | "
                )
            if self.writer:
                scalars = {"training": loss}
                if val_loss is not None:
                    scalars.update({"validation": val_loss})
                # Records both losses for each epoch under the main tag "loss"
                self.writer.add_scalars(
                    main_tag="loss",
                    tag_scalar_dict=scalars,
                    global_step=epoch,
                )
        end_time = timer()
        console.log(
            f"[bold][cyan][INFO][/][/] Total training time: {end_time-start_time:.3f} seconds"
        )
        self.save_checkpoint(epoch, LATEST=True)
        if self.writer:
            self.writer.close()

    def save_checkpoint(self, epoch, LATEST: bool = False):

        ckpt_dir_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "checkpoints",
        )
        os.makedirs(ckpt_dir_path, exist_ok=True)

        # Builds dictionary with all elements for resuming training
        checkpoint = {
            "epoch": self.total_epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.losses,
            "val_loss": self.val_losses,
        }
        checkpoint_name = (
            "epoch_{ckpt_no}.pth".format(
                ckpt_no=epoch,
            )
            if not LATEST
            else "latest.pth"
        )
        filename = os.path.join(ckpt_dir_path, checkpoint_name)
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        self.is_load_checkpoint = True
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_epochs = checkpoint["epoch"]
        self.losses = checkpoint["loss"]
        self.val_losses = checkpoint["val_loss"]

        self.model.train()

    def predict(self, x):
        self.model.eval()
        x_tensor = torch.as_tensor(x).float()
        y_hat_tensor = self.model(x_tensor.to(self.device))
        self.model.train()
        return y_hat_tensor.detach().cpu().numpy()

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label="Training Loss", c="b")
        if self.val_loader:
            plt.plot(self.val_losses, label="Validation Loss", c="r")
        plt.yscale("log")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(self.parent_folder_path, "plot_losses"))
        return fig

    def add_graph(self):
        if self.train_loader and self.writer:
            # Fetches a single mini-batch so we can use add_graph
            x_sample, y_sample = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_sample.to(self.device))

    def count_parameters(self):
        return sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )  # noqa

    @staticmethod
    def _visualize_tensors(axs, x, y=None, yhat=None, layer_name="", title=None):
        # The number of images is the number of subplots in a row
        n_images = len(axs)
        # Gets max and min values for scaling the grayscale
        minv, maxv = np.min(x[:n_images]), np.max(x[:n_images])
        # For each image
        for j, image in enumerate(x[:n_images]):
            ax = axs[j]
            # Sets title, labels, and removes ticks
            if title is not None:
                ax.set_title(f"{title} #{j}", fontsize=12)
            shp = np.atleast_2d(image).shape
            ax.set_ylabel(f"{layer_name}\n{shp[0]}x{shp[1]}", rotation=0, labelpad=40)
            xlabel1 = "" if y is None else f"\nLabel: {y[j]}"
            xlabel2 = "" if yhat is None else f"\nPredicted: {yhat[j]}"
            xlabel = f"{xlabel1}{xlabel2}"
            if len(xlabel):
                ax.set_xlabel(xlabel, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

            # Plots weight as an image
            ax.imshow(np.atleast_2d(image.squeeze()), cmap="gray", vmin=minv, vmax=maxv)
            ax.savefig("visualize_filters.png")
        return

    def visualize_filters(self, layer_name, **kwargs):
        try:
            # Gets the layer object from the model
            layer = self.model
            for name in layer_name.split("."):
                layer = getattr(layer, name)
            # We are only looking at filters for 2D convolutions
            if isinstance(layer, nn.Conv2d):
                # Takes the weight information
                weights = layer.weight.data.cpu().numpy()
                # weights -> (channels_out (filter), channels_in, H, W)
                n_filters, n_channels, _, _ = weights.shape

                # Builds a figure
                size = (2 * n_channels + 2, 2 * n_filters)
                fig, axes = plt.subplots(n_filters, n_channels, figsize=size)
                axes = np.atleast_2d(axes)
                axes = axes.reshape(n_filters, n_channels)
                # For each channel_out (filter)
                for i in range(n_filters):
                    Engine._visualize_tensors(
                        axes[i, :],
                        weights[i],
                        layer_name=f"Filter #{i}",
                        title="Channel",
                    )

                for ax in axes.flat:
                    ax.label_outer()

                fig.tight_layout()
                fig.savefig("visualize_filters.png")
        except AttributeError:
            return
