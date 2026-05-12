"""Train RecVAE, probe latents with logistic regression vs PCA baseline.

Variation from canonical
------------------------
- Trains the canonical RecVAE on a small synthetic cohort, then extracts a
  per-subject latent vector (the mean of h_t over time).
- Fits a CN-vs-AD logistic classifier on those latents and compares to a
  PCA baseline computed on the flattened raw volumes.
- Both classifiers report train AND held-out test accuracy. With N=8 the
  numbers are very noisy: the point is to demonstrate the *pattern*, not
  to publish accuracy figures.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from recvae import (  # noqa: E402
    Config,
    FMRIDataset,
    RecVAEModel,
    build_dataloader,
    fit,
    set_seed,
    synthetic_cohort,
)

# Make the sklearn dependency explicit. We probe once at import time so the
# warning is visible at the top of the script's output, rather than buried
# inside fit_logistic() where a first-time learner copy-pasting `from
# sklearn.linear_model import LogisticRegression` would hit ImportError
# without any context. The torch-fallback path remains the same.
try:
    import sklearn  # noqa: F401, E402

    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False
    print(
        "WARNING: scikit-learn not installed. Falling back to a torch-rolled logistic.\n"
        'To install:  pip install -e ".[examples]"',
    )


def extract_latents(model: RecVAEModel, volumes: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
    """Run a frozen forward pass and average h_t over time."""
    model.eval()
    with torch.no_grad():
        which = torch.arange(volumes.shape[0])
        h_batch = h0.expand(volumes.shape[0], -1)
        out = model(volumes, h_batch, which)
        return out.h.mean(dim=1)  # (N, latent_dim)


def pca_features(train_x: torch.Tensor, test_x: torch.Tensor, q: int = 8):
    """8-component PCA on flattened volumes (fit on train, applied to both)."""
    Xtr = train_x.reshape(train_x.shape[0], -1).float()
    Xte = test_x.reshape(test_x.shape[0], -1).float()
    mean = Xtr.mean(dim=0, keepdim=True)
    U, S, V = torch.svd_lowrank(Xtr - mean, q=min(q, Xtr.shape[0] - 1))
    return (Xtr - mean) @ V, (Xte - mean) @ V


def fit_logistic(feat_train: torch.Tensor, y_train: torch.Tensor,
                 feat_test: torch.Tensor, y_test: torch.Tensor):
    """Use sklearn if available; otherwise fall back to a torch logistic.

    Availability is decided once at import time (see ``_HAVE_SKLEARN``) so
    the warning surfaces at the top of the script rather than inside this
    function.
    """
    if _HAVE_SKLEARN:
        from sklearn.linear_model import LogisticRegression  # noqa: WPS433
        clf = LogisticRegression(max_iter=200)
        clf.fit(feat_train.numpy(), y_train.numpy())
        return clf.score(feat_train.numpy(), y_train.numpy()), \
               clf.score(feat_test.numpy(), y_test.numpy()), "sklearn"
    # Fallback: 100 Adam steps on nn.Linear + BCE.
    torch.manual_seed(0)
    D = feat_train.shape[1]
    head = torch.nn.Linear(D, 2)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(100):
        opt.zero_grad()
        loss_fn(head(feat_train), y_train).backward()
        opt.step()
    with torch.no_grad():
        tr_acc = (head(feat_train).argmax(1) == y_train).float().mean().item()
        te_acc = (head(feat_test).argmax(1) == y_test).float().mean().item()
    return tr_acc, te_acc, "torch-fallback"


def main() -> int:
    set_seed(2022)

    # Train cohort: 4 CN + 4 AD; test cohort: 2 CN + 2 AD with a different seed.
    train_x, train_y = synthetic_cohort(n_cn=4, n_ad=4, T=4, seed=2022)
    test_x, test_y = synthetic_cohort(n_cn=2, n_ad=2, T=4, seed=4242)

    cfg = Config(tol_time=4, epochs=3, batch_size=2, learning_rate=1e-5)
    model = RecVAEModel(train_size=train_x.shape[0], cfg=cfg)
    dl = build_dataloader(FMRIDataset(train_x), batch_size=2, shuffle=True, seed=2022)
    h0 = torch.zeros(1, cfg.latent_dim)
    fit(model, dl, h0, cfg=cfg, epochs=cfg.epochs)

    # ---- RecVAE latent probe ----
    z_train = extract_latents(model, train_x, h0)
    z_test = extract_latents(model, test_x, h0)
    rec_tr, rec_te, backend = fit_logistic(z_train, train_y, z_test, test_y)

    # ---- PCA baseline ----
    pca_train, pca_test = pca_features(train_x, test_x, q=8)
    pca_tr, pca_te, _ = fit_logistic(pca_train, train_y, pca_test, test_y)

    print(f"logistic backend           : {backend}")
    print(f"RecVAE latents  train_acc  : {rec_tr:.2f}")
    print(f"RecVAE latents  test_acc   : {rec_te:.2f}")
    print(f"PCA  baseline   train_acc  : {pca_tr:.2f}")
    print(f"PCA  baseline   test_acc   : {pca_te:.2f}")
    print("caveat: N=8 train / N=4 test, accuracies are noisy "
          "and not a real evaluation of the representation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
