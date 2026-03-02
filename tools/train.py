import os

import torch
from torch.utils.data import DataLoader

from src.constants import CKPT_FOLDER


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    n_epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    val_loader: DataLoader = None,
    ckpt_prefix: str = "",
    lr_scheduler=None,
    use_mixed_precision=False,
    autocast_dtype=torch.float16,
    device="cpu",
):
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    scaler = torch.amp.GradScaler(
        enabled=(use_mixed_precision and autocast_dtype == torch.float16)
    )

    device_type_str = device.type if isinstance(device, torch.device) else device

    for epoch in range(n_epochs):
        train_loss = 0

        model.train()
        for data, label in train_loader:
            if label.dim() == 1:
                label = label.unsqueeze(1)

            data = data.to(device=device)
            label = label.to(device=device)

            optimizer.zero_grad()

            with torch.autocast(
                device_type=device_type_str,
                dtype=autocast_dtype,
                enabled=use_mixed_precision,
            ):
                output = model(data)
                loss = criterion(output, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * data.size(0)

        if lr_scheduler is not None:
            lr_scheduler.step()

        train_loss /= len(train_loader.sampler)
        train_losses.append(train_loss)

        epoch_losses = {"train_loss": train_loss}
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, label in val_loader:
                    if label.dim() == 1:
                        label = label.unsqueeze(1)

                    data = data.to(device=device)
                    label = label.to(device=device)

                    with torch.autocast(
                        device_type=device_type_str,
                        dtype=autocast_dtype,
                        enabled=use_mixed_precision,
                    ):
                        output = model(data)
                        loss = criterion(output, label)

                    val_loss += loss.item() * data.size(0)

            val_loss /= len(val_loader.sampler)
            val_losses.append(val_loss)
            epoch_losses["val_loss"] = val_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(CKPT_FOLDER, ckpt_prefix + "best_model.pt"),
                )
                print("Model saved !")

        formatted_str = f"Epoch: {epoch + 1} | Losses: {{"
        if "train_loss" in epoch_losses:
            formatted_str += f"train_loss: {epoch_losses['train_loss']:.5f}, "
        if "val_loss" in epoch_losses:
            formatted_str += f"val_loss: {epoch_losses['val_loss']:.5f}"
        formatted_str += "}"
        print(formatted_str)

    torch.save(
        model.state_dict(), os.path.join(CKPT_FOLDER, ckpt_prefix + "final_model.pt")
    )
    torch.cuda.empty_cache()

    return train_losses, val_losses
