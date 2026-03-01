import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.generate import generate_Z


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
            H=H[i], eta=1, N_paths=paths_per_H, n_points=n_points, T=T
        )

    paths = torch.Tensor(paths).to(device=device)

    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

    plt.figure(figsize=(10, 10))
    plt.plot(H, H, color="tomato", linestyle="--", linewidth=2, label="H réel")

    for idx, model in enumerate(models):
        model.to(device=device)
        model.eval()
        predictions = []

        print(f"Predicting with model {idx + 1} ({model_labels[idx]})...")
        for i in tqdm(range(n_H)):
            with torch.no_grad():
                try:
                    current_predictions = model(
                        paths[i * paths_per_H : (i + 1) * paths_per_H, :].unsqueeze(1)
                    ).ravel()
                except:
                    current_predictions = model(
                        paths[i * paths_per_H : (i + 1) * paths_per_H, :]
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

    plt.title("H prédit vs. H réel")
    plt.xlabel("H")
    plt.ylabel("H prédit")
    plt.grid(True)
    plt.legend()
    plt.show()
