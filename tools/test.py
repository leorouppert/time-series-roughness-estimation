import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.generate import generate_Z
from src.models import LSTM


def compute_mae(model, test_loader, device="cpu"):
    model.eval()
    with torch.no_grad():
        mae_sum, total = 0, 0

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).ravel()

            total += labels.size(0)
            mae_sum += torch.abs(outputs - labels).sum().item()

        mae = mae_sum / total
        print(f"Model Test MAE: {mae:.4f}")

        return mae


def sequential_pred(model, raw_data, inference_batch_size=256, device="cpu"):
    model.eval()

    dataset = TensorDataset(raw_data)
    dataloader = DataLoader(dataset, batch_size=inference_batch_size)

    with torch.no_grad():
        preds = torch.Tensor([])

        for inputs in dataloader:
            inputs = inputs[0].to(device)
            outputs = model(inputs).ravel().cpu()

            preds = torch.cat((preds, outputs))

    return preds


def draw_predictions(
    models,
    model_labels,
    paths_per_H,
    n_points,
    T,
    H_min=0.0,
    H_max=1.0,
    n_H=11,
    device="cpu",
):

    H = np.linspace(H_min, H_max, n_H)
    paths = np.zeros((n_H * paths_per_H, n_points))

    print("Generating paths ...")
    for i in tqdm(range(n_H)):
        paths[paths_per_H * i : paths_per_H * (i + 1), :] = generate_Z(
            H=H[i], eta=1, n_paths=paths_per_H, n_points=n_points, T=T
        )

    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

    plt.figure(figsize=(10, 10))
    plt.plot(H, H, color="tomato", linestyle="--", linewidth=2, label="Ground Truth H")

    for idx, model in enumerate(models):
        model.to(device=device)
        model.eval()
        predictions = []

        print(f"Predicting with model {idx + 1} ({model_labels[idx]})...")
        for i in tqdm(range(n_H)):
            if isinstance(model, LSTM):
                current_predictions = sequential_pred(
                    model,
                    paths[i * paths_per_H : (i + 1) * paths_per_H, :],
                    device=device,
                )
            else:
                with torch.no_grad():
                    current_predictions = model(
                        paths[i * paths_per_H : (i + 1) * paths_per_H, :].to(
                            device=device
                        )
                    ).ravel()

            predictions.append(current_predictions.detach().cpu().numpy())

        mean_predictions = np.mean(predictions, axis=1)
        std_predictions = np.std(predictions, axis=1)

        plt.errorbar(
            x=H,
            y=mean_predictions,
            yerr=std_predictions,
            capsize=5,
            color=colors[idx],
            fmt="-",
            label=model_labels[idx],
        )

    plt.title("Predicted H vs. ground truth")
    plt.xlabel("Ground truth")
    plt.ylabel("Predicted H")
    plt.grid(True)
    plt.legend()
    plt.show()


def ou_test(model, ou_paths, device="cpu"):
    model.eval()
    ou_paths = ou_paths.to(device=device)

    if isinstance(model, LSTM):
        predictions_ou = (
            sequential_pred(model, ou_paths, device=device).cpu().numpy().ravel()
        )
    else:
        with torch.no_grad():
            predictions_ou = model(ou_paths).cpu().numpy().ravel()

    mean = np.mean(predictions_ou)
    median = np.median(predictions_ou)
    lower = np.percentile(predictions_ou, 2.5)
    upper = np.percentile(predictions_ou, 97.5)

    plt.figure(figsize=(10, 6))
    plt.hist(
        predictions_ou,
        bins=30,
        alpha=0.5,
        color="blue",
        edgecolor="black",
        density=False,
    )
    plt.axvline(mean, color="red", linestyle="--", label=f"Mean: {mean:.4f}")
    plt.axvline(median, color="green", linestyle="-", label=f"Median: {median:.4f}")
    plt.axvspan(
        lower,
        upper,
        color="gray",
        alpha=0.3,
        label=f"95% CI: [{lower:.4f}, {upper:.4f}]",
    )
    plt.legend()
    plt.title("Distribution of predictions")
    plt.xlabel("Prediction Values")
    plt.ylabel("Count")
    plt.show()
