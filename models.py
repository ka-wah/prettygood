from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Callable, Optional

import numpy as np

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer

# LightGBM (optional)
try:
    import lightgbm as lgb
    LGB_OK = True
except Exception:
    LGB_OK = False

# Torch (optional)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
except Exception:
    TORCH_OK = False


class MeanEnsemble:
    """
    Equal-weight ensemble over already-fitted base regressors.
    Optionally apply an x_transform(X) before predicting, so members trained on
    a transformed/augmented feature space still get the right shape.
    """
    def __init__(self, members: dict[str, object],
                 x_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.members = members
        self.fitted_ = True
        self.n_features_in_ = None
        self.x_transform = x_transform

    def fit(self, X, y=None):
        X = np.asarray(X)
        X_use = self.x_transform(X) if (self.x_transform is not None and X.ndim == 2) else X
        self.n_features_in_ = X_use.shape[1] if X_use is not None and X_use.ndim == 2 else None
        self.fitted_ = True
        return self

    def predict(self, X):
        X = np.asarray(X)
        X_use = self.x_transform(X) if self.x_transform is not None else X
        preds = [m.predict(X_use) for m in self.members.values()]
        return np.mean(np.column_stack(preds), axis=1)


def build_equal_weight_ensemble(names: List[str], fitted: Dict[str, Any],
                                x_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None):
    members = {n: fitted[n] for n in names}
    return MeanEnsemble(members, x_transform=x_transform)


@dataclass
class ModelSpec:
    name: str
    factory: Callable[[], Any]
    grid_default: Dict[str, List[Any]]
    kind: str = "generic"  # "linear" | "tree" | "nn"


# -----------------------
# Torch FFN (sklearn API)
# -----------------------
class TorchFFNRegressor:
    def __init__(self,
                 hidden_layer_sizes=(128, 64),
                 activation="relu",
                 dropout=0.0,
                 learning_rate=1e-3,
                 weight_decay=1e-4,
                 max_epochs=300,
                 batch_size=256,
                 patience=10,
                 val_split=0.1,
                 seed=0,
                 device="auto",
                 n_hidden_layers=None,
                 hidden_width=None,
                 width_shrink=1.0,
                 loss="mse",
                 huber_delta=1.0):
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.val_split = float(val_split)
        self.seed = int(seed)
        self.device = device
        self._model = None
        self._fitted = False
        self._in_dim = None
        self.n_hidden_layers = None if n_hidden_layers is None else int(n_hidden_layers)
        self.hidden_width = None if hidden_width is None else int(hidden_width)
        self.width_shrink = float(width_shrink)
        self.loss = str(loss)
        self.huber_delta = float(huber_delta)

    def get_params(self, deep=True):
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "activation": self.activation,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "val_split": self.val_split,
            "seed": self.seed,
            "device": self.device,
            "n_hidden_layers": self.n_hidden_layers,
            "hidden_width": self.hidden_width,
            "width_shrink": self.width_shrink,
            "loss": self.loss,
            "huber_delta": self.huber_delta,
        }

    def set_params(self, **params):
        int_keys = {"max_epochs", "batch_size", "patience", "seed",
                    "n_hidden_layers", "hidden_width"}
        float_keys = {"dropout", "learning_rate", "weight_decay",
                      "val_split", "width_shrink", "huber_delta"}
        for k, v in params.items():
            if hasattr(self, k):
                if k in int_keys:
                    setattr(self, k, int(v))
                elif k in float_keys:
                    setattr(self, k, float(v))
                elif k == "hidden_layer_sizes":
                    if isinstance(v, (list, tuple)):
                        setattr(self, k, tuple(int(x) for x in v))
                    else:
                        setattr(self, k, (int(v),))
                else:
                    setattr(self, k, v)
        return self

    def _resolved_sizes(self):
        # If no layer-count / width overrides, and explicit sizes exist, use them.
        if (self.n_hidden_layers is None) and (self.hidden_width is None) \
           and self.hidden_layer_sizes and len(self.hidden_layer_sizes) > 0:
            return tuple(int(h) for h in self.hidden_layer_sizes)

        # Else build sizes from (n_hidden_layers, hidden_width, width_shrink)
        L = self.n_hidden_layers if (self.n_hidden_layers is not None) else 1
        W = self.hidden_width if (self.hidden_width is not None) else 64
        r = self.width_shrink if (self.width_shrink is not None) else 1.0
        sizes = [max(1, int(round(W * (r ** i)))) for i in range(L)]
        return tuple(sizes)


    def _act(self):
        a = str(self.activation).lower()
        if a == "relu":
            return nn.ReLU()
        if a == "tanh":
            return nn.Tanh()
        if a == "gelu":
            return nn.GELU()
        return nn.ReLU()

    def _build(self, in_dim: int):
        sizes = self._resolved_sizes()
        layers = []
        last = in_dim
        for h in sizes:
            layers += [nn.Linear(last, int(h)), self._act()]
            if self.dropout and self.dropout > 0:
                layers.append(nn.Dropout(p=float(self.dropout)))
            last = int(h)
        layers.append(nn.Linear(last, 1))
        return nn.Sequential(*layers)

    def _device(self):
        if self.device == "cpu":
            return torch.device("cpu")
        if self.device == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _set_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)

    def fit(self, X, y, sample_weight=None):
        if not TORCH_OK:
            raise RuntimeError("TorchFFNRegressor requires PyTorch to be installed.")

        import torch
        import torch.nn.functional as F

        # --- inputs & basic state ---
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        n = X.shape[0]
        self._in_dim = X.shape[1]
        self._set_seed()
        device = self._device()

        # --- weights (optional) ---
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
            if w.shape[0] != n:
                raise ValueError(f"sample_weight has length {w.shape[0]} but X has {n} rows.")
            # Normalise to mean ~ 1 to keep the loss scale stable
            m = float(np.nanmean(w))
            w = np.where(np.isfinite(w), w, 0.0)
            w = w / (m if (m and np.isfinite(m)) else 1.0)
        else:
            w = None

        # --- train/val split for early stopping ---
        nv = int(max(1, min(int(n * self.val_split), n - 1))) if self.patience > 0 and self.val_split > 0 else 0
        if nv > 0:
            idx = np.random.permutation(n)
            X_tr, y_tr = X[idx[nv:]], y[idx[nv:]]
            X_va, y_va = X[idx[:nv]], y[idx[:nv]]
            if w is not None:
                w_tr = w[idx[nv:]]
                w_va = w[idx[:nv]]
            else:
                w_tr = w_va = None
        else:
            X_tr, y_tr = X, y
            X_va, y_va = None, None
            w_tr = w
            w_va = None

        # --- model/optim ---
        model = self._build(self._in_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # choose loss
        loss_mode = str(self.loss).lower()
        if loss_mode in {"huber", "smooth_l1"}:
            delta = float(self.huber_delta)

            def loss_unreduced(pred, target):
                return F.huber_loss(pred, target, delta=delta, reduction="none")
        else:
            mse = torch.nn.MSELoss(reduction="none")

            def loss_unreduced(pred, target):
                return mse(pred, target)

        # Build DataLoader; include weights if provided
        if w_tr is None:
            train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        else:
            train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr),
                                     torch.from_numpy(w_tr))
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        best_loss = float("inf")
        best_state = None
        no_improve = 0

        for _ in range(self.max_epochs):
            model.train()
            for batch in train_dl:
                if w_tr is None:
                    xb, yb = batch
                    wb = None
                else:
                    xb, yb, wb = batch
                xb = xb.to(device)
                yb = yb.to(device)
                raw = loss_unreduced(model(xb), yb)

                if wb is None:
                    loss_val = raw.mean()
                else:
                    wb = wb.to(device).view(-1, 1)
                    loss_val = (raw * wb).mean()

                opt.zero_grad()
                loss_val.backward()
                opt.step()

            # --- validation (for early stopping) ---
            if X_va is not None:
                model.eval()
                with torch.no_grad():
                    xb = torch.from_numpy(X_va).to(device)
                    yb = torch.from_numpy(y_va).to(device)
                    raw = loss_unreduced(model(xb), yb)
                    if w_va is None:
                        val_loss = raw.mean().item()
                    else:
                        wb = torch.from_numpy(w_va).to(device).view(-1, 1)
                        val_loss = (raw * wb).mean().item()

                if val_loss + 1e-12 < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if self.patience > 0 and no_improve >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        self._model = model.to("cpu")
        self._fitted = True
        return self

    def predict(self, X):
        assert self._fitted, "Model not fitted."
        X = np.asarray(X, dtype=np.float32)
        import torch
        with torch.no_grad():
            xb = torch.from_numpy(X)
            yhat = self._model(xb).numpy().reshape(-1)
        return yhat


# -----------------------
# Specs / registry
# -----------------------
def _ols_spec() -> ModelSpec:
    return ModelSpec(
        name="ols",
        factory=lambda: LinearRegression(),
        grid_default={"fit_intercept": [True]},
        kind="linear",
    )


def _ridge_spec() -> ModelSpec:
    return ModelSpec(
        name="ridge",
        factory=lambda: Ridge(random_state=0),
        grid_default={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        kind="linear",
    )


def _lasso_spec() -> ModelSpec:
    return ModelSpec(
        name="lasso",
        factory=lambda: Lasso(random_state=0, max_iter=10000),
        grid_default={"alpha": [0.0001, 0.001, 0.01, 0.1]},
        kind="linear",
    )


def _enet_spec() -> ModelSpec:
    return ModelSpec(
        name="elasticnet",
        factory=lambda: ElasticNet(random_state=0, max_iter=10000),
        grid_default={"alpha": [0.0001, 0.001, 0.01, 0.1], "l1_ratio": [0.2, 0.5, 0.8]},
        kind="linear",
    )


def _pcr_spec() -> ModelSpec:
    return ModelSpec(
        name="pcr",
        factory=lambda: Pipeline(
            [("scale", StandardScaler(with_mean=False)), ("pca", PCA(random_state=0)), ("ridge", Ridge(random_state=0))]
        ),
        grid_default={"pca__n_components": [2, 3, 4, 5, 6], "ridge__alpha": [0.1, 1.0, 10.0]},
        kind="linear",
    )


def _pls_spec() -> ModelSpec:
    return ModelSpec(
        name="pls",
        factory=lambda: PLSRegression(),
        grid_default={"n_components": [1, 2, 3, 4, 5, 6]},
        kind="linear",
    )


def _rf_spec() -> ModelSpec:
    return ModelSpec(
        name="rf",
        factory=lambda: RandomForestRegressor(random_state=0, n_jobs=-1),
        grid_default={
            "n_estimators": [300, 600, 900],
            "max_depth": [5, 8, 10],
            "max_features": ["sqrt", 0.3, 0.5],
            "min_samples_leaf": [5, 10, 20],
            "bootstrap": [True],
        },
        kind="tree",
    )


def _gbr_like_specs() -> List[ModelSpec]:
    if LGB_OK:
        base_grid = {
            "n_estimators": [400, 800, 1200],
            "learning_rate": [0.02, 0.05],
            "max_depth": [3, 5, 7],
            "num_leaves": [15, 31, 63],
            "min_data_in_leaf": [20, 50, 100, 200, 300],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0.0, 0.5, 1.0],
            "reg_lambda": [0.0, 0.5, 1.0],
        }
        return [
            ModelSpec(
                name="lgbm_gbdt",
                factory=lambda: lgb.LGBMRegressor(boosting_type="gbdt", objective="regression", random_state=0),
                grid_default={**base_grid, "metric": ["l1"], "verbosity": [-1]},
                kind="tree",
            ),
            ModelSpec(
                name="lgbm_gbdt_huber",
                factory=lambda: lgb.LGBMRegressor(boosting_type="gbdt", objective="huber", random_state=0),
                grid_default={**base_grid,
                              "huber_delta": [0.5, 1.0, 2.0],
                              "metric": ["huber"],
                              "verbosity": [-1]},
                kind="tree",
            ),
            ModelSpec(
                name="lgbm_dart",
                factory=lambda: lgb.LGBMRegressor(boosting_type="dart", objective="regression", random_state=0),
                grid_default={**base_grid,
                              "drop_rate": [0.05, 0.10, 0.20],
                              "max_drop": [10, 50],
                              "skip_drop": [0.25, 0.5],
                              "metric": ["l1"],
                              "verbosity": [-1]},
                kind="tree",
            ),
            ModelSpec(
                name="lgbm_dart_huber",
                factory=lambda: lgb.LGBMRegressor(boosting_type="dart", objective="huber", random_state=0),
                grid_default={**base_grid,
                              "drop_rate": [0.05, 0.10, 0.20],
                              "max_drop": [10, 50],
                              "skip_drop": [0.25, 0.5],
                              "huber_delta": [0.5, 1.0, 2.0],
                              "metric": ["huber"],
                              "verbosity": [-1]},
                kind="tree",
            ),
        ]
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        base_grid = {
            "n_estimators": [300, 600, 1000],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.05, 0.1],
        }
        return [
            ModelSpec(
                name="gbr",
                factory=lambda: GradientBoostingRegressor(random_state=0),
                grid_default=base_grid,
                kind="tree",
            ),
            ModelSpec(
                name="gbr_huber",
                factory=lambda: GradientBoostingRegressor(loss="huber", random_state=0),
                grid_default={**base_grid, "alpha": [0.85, 0.9, 0.95]},
                kind="tree",
            ),
        ]


def _ffn_spec() -> ModelSpec:
    if TORCH_OK:
        return ModelSpec(
            name="ffn",
            factory=lambda: TorchFFNRegressor(),
            grid_default={
                "hidden_layer_sizes": [(64,), (128,), (128, 64)],
                "activation": ["relu", "tanh", "gelu"],
                "dropout": [0.0, 0.1, 0.2],
                "learning_rate": [1e-3, 5e-4, 1e-4],
                "weight_decay": [1e-6, 1e-5, 1e-4],
                "max_epochs": [200, 300],
                "batch_size": [128, 256, 512],
                "patience": [10, 20],
                "val_split": [0.1],
                "seed": [0],
                "device": ["auto"],
                "loss": ["mse", "huber"],
                "huber_delta": [0.5, 1.0, 2.0],
            },
            kind="nn",
        )
    else:
        return ModelSpec(
            name="ffn",
            factory=lambda: MLPRegressor(random_state=0, max_iter=300),
            grid_default={
                "hidden_layer_sizes": [(64,), (128,), (128, 64)],
                "alpha": [1e-5, 1e-4, 1e-3],
                "learning_rate_init": [1e-3, 5e-4, 1e-4],
            },
            kind="nn",
        )


# -----------------------
# Registry builder
# -----------------------
def _yaml_space(space: Dict[str, Any]) -> Dict[str, List[Any]]:
    # passthrough; sampler interprets {values:...} or {dist:...,low,high}
    return space


def registry_from_yaml(cfg: Dict[str, Any]) -> Dict[str, ModelSpec]:
    spaces = cfg.get("tuning", {}).get("spaces", {}) if cfg else {}
    reg: Dict[str, ModelSpec] = {}
    for s in [_ols_spec(), _ridge_spec(), _lasso_spec(), _enet_spec(), _pcr_spec(), _pls_spec()]:
        ms = s
        if ms.name in spaces:
            ms = ModelSpec(ms.name, ms.factory, _yaml_space(spaces[ms.name]), kind=ms.kind)
        reg[ms.name] = ms
    for s in [_rf_spec(), *_gbr_like_specs(), _ffn_spec()]:
        ms = s
        if ms.name in spaces:
            ms = ModelSpec(ms.name, ms.factory, _yaml_space(spaces[ms.name]), kind=ms.kind)
        reg[ms.name] = ms

    # --- Make FFN self-contained: Impute → Yeo–Johnson → Standardize ---
    if "ffn" in reg:
        base_spec = reg["ffn"]
        base_factory = base_spec.factory
        base_grid = dict(getattr(base_spec, "grid_default", {}) or {})

        def ffn_with_prep():
            base_ffn = base_factory()  # original FFN estimator (Torch or sklearn)
            return Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("yeojohnson", PowerTransformer(method="yeo-johnson", standardize=False)),
                ("scale", StandardScaler(with_mean=True, with_std=True)),
                ("ffn", base_ffn),
            ])

        # Prefix existing FFN grid keys so tuning still works (e.g., ffn__learning_rate)
        prefixed_grid = {f"ffn__{k}": v for k, v in base_grid.items()}

        # Replace spec in-place
        reg["ffn"] = ModelSpec(
            name=base_spec.name,
            factory=ffn_with_prep,
            grid_default=prefixed_grid,
            kind=base_spec.kind,
        )

    return reg
