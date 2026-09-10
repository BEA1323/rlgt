# dry_run.py

import os
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.seed import seed_everything

from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

from redelex import datasets as ctu_datasets
from redelex import tasks as ctu_tasks

from agg_features import precompute_agg_features
from model import RelGT
from utils import GloveTextEmbedding, RelGTTokens


# ============================================================
# CONFIG
# ============================================================

DATASET = "rel-f1"
TASK = "driver-top3"

BATCH_SIZE = 4
NUM_NEIGHBORS = 300
NUM_WORKERS = 1
SEED = 42

CACHE_DIR = os.path.expanduser("~/.cache/relbench_examples")

DEVICE = torch.device("cuda:0")

# ============================================================
# SETUP
# ============================================================

seed_everything(SEED)

assert torch.cuda.is_available(), "CUDA is not available!"

torch.cuda.set_device(DEVICE)

print("=" * 70)
print("ONE-BATCH DRY RUN")
print("=" * 70)
print(f"Dataset : {DATASET}")
print(f"Task    : {TASK}")
print(f"Device  : {DEVICE}")


# ============================================================
# LOAD DATASET / TASK
# ============================================================

print("\n[1/9] Loading dataset and task...")

if DATASET.startswith("ctu-"):
    dataset: Dataset = get_dataset(DATASET, download=False)
    task: EntityTask = get_task(DATASET, TASK, download=False)
else:
    dataset: Dataset = get_dataset(DATASET, download=True)
    task: EntityTask = get_task(DATASET, TASK, download=True)

print("✓ Dataset/task loaded")


# ============================================================
# STOCHASTIC TYPES
# ============================================================

print("\n[2/9] Preparing column types...")

stypes_cache_path = Path(
    f"{CACHE_DIR}/{DATASET}/stypes.json"
)

try:
    with open(stypes_cache_path, "r") as f:
        col_to_stype_dict = json.load(f)

    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)

except FileNotFoundError:

    col_to_stype_dict = get_stype_proposal(
        dataset.get_db()
    )

    stypes_cache_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(stypes_cache_path, "w") as f:
        json.dump(
            col_to_stype_dict,
            f,
            indent=2,
            default=str
        )

print("✓ Column types ready")


# ============================================================
# BUILD GRAPH
# ============================================================

print("\n[3/9] Building graph...")

data_graph, col_stats_dict = make_pkey_fkey_graph(
    dataset.get_db(),

    col_to_stype_dict=col_to_stype_dict,

    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(
            device="cuda:0"
        ),
        batch_size=256,
    ),

    cache_dir=(
        f"{CACHE_DIR}/{DATASET}/materialized"
    ),
)

print("✓ Graph built")


# ============================================================
# AGGREGATION FEATURES
# ============================================================

print("\n[4/9] Checking aggregation features...")

agg_cache_path = (
    f"{CACHE_DIR}/{DATASET}/agg_features.h5"
)

if not os.path.exists(agg_cache_path):

    print("Aggregation cache does not exist.")
    print("Precomputing aggregation features...")

    precompute_agg_features(
        dataset=DATASET,
        db=dataset.get_db(),
        max_depth=2,
        out_path=agg_cache_path,
    )

    print("✓ Aggregation features computed")

else:

    print(
        f"✓ Using existing aggregation cache:\n"
        f"  {agg_cache_path}"
    )


# ============================================================
# CREATE RelGTTokens
# ============================================================

print("\n[5/9] Creating RelGTTokens...")

train_data = RelGTTokens(
    data=data_graph,
    task=task,

    K=NUM_NEIGHBORS,

    split="train",

    undirected=True,

    precompute=True,

    precomputed_dir=(
        f"{CACHE_DIR}/precomputed/"
        f"{DATASET}/{TASK}"
    ),

    agg_precomputed_path=agg_cache_path,

    num_workers=NUM_WORKERS,

    train_stage="finetune",
)

assert train_data.agg_dims is not None, (
    "agg_dims is None. "
    "Aggregation features were not loaded."
)

print("✓ RelGTTokens created")

print("\nAggregation dimensions:")

for node_type, dim in train_data.agg_dims.items():
    print(f"  {node_type}: {dim}")


# ============================================================
# CREATE DATALOADER
# ============================================================

print("\n[6/9] Creating DataLoader...")

from torch.utils.data import DataLoader

loader_train = DataLoader(
    train_data,

    batch_size=BATCH_SIZE,

    shuffle=False,

    collate_fn=train_data.collate,

    num_workers=NUM_WORKERS,

    pin_memory=True,
)

print("✓ DataLoader created")


# ============================================================
# TASK / LOSS
# ============================================================

if task.task_type == TaskType.BINARY_CLASSIFICATION:

    out_channels = 1
    loss_fn = BCEWithLogitsLoss()

elif task.task_type == TaskType.REGRESSION:

    out_channels = 1
    loss_fn = L1Loss()

elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:

    out_channels = task.num_labels
    loss_fn = BCEWithLogitsLoss()

else:

    raise ValueError(
        f"Unsupported task type: {task.task_type}"
    )


# ============================================================
# CREATE MODEL
# ============================================================

print("\n[7/9] Creating model...")

# These match the arguments in your training script.
class Args:
    num_layers = 1
    channels = 512
    num_heads = 4
    ff_dropout = 0.1
    attn_dropout = 0.1
    gt_conv_type = "full"
    ablate = "none"
    gnn_pe_dim = 0
    num_centroids = 4096
    num_neighbors = 300


args = Args()

model = RelGT(
    num_nodes=train_data.data.num_nodes,

    max_neighbor_hop=train_data.max_neighbor_hop,

    node_type_map=train_data.node_type_to_index,

    col_names_dict={
        node_type:
        train_data.data[node_type].tf.col_names_dict
        for node_type in train_data.data.node_types
    },

    col_stats_dict=col_stats_dict,

    local_num_layers=args.num_layers,

    channels=args.channels,

    out_channels=out_channels,

    global_dim=args.channels // 2,

    heads=args.num_heads,

    ff_dropout=args.ff_dropout,

    attn_dropout=args.attn_dropout,

    conv_type=args.gt_conv_type,

    ablate=args.ablate,

    gnn_pe_dim=args.gnn_pe_dim,

    num_centroids=args.num_centroids,

    sample_node_len=args.num_neighbors,

    agg_dim_dict=train_data.agg_dims,

    feature_meta=train_data.feature_meta,

    args=args,
).to(DEVICE)


# Match your original int16 handling
for name, param in model.named_parameters():

    if param.dtype == torch.int16:
        param.data = param.data.to(torch.int64)


for name, buf in model.named_buffers():

    if buf.dtype == torch.int16:
        buf.data = buf.data.to(torch.int64)


print("✓ Model created")


# ============================================================
# GET ONE BATCH
# ============================================================

print("\n[8/9] Getting one batch...")

batch = next(iter(loader_train))

print("✓ Batch obtained")

print("\nBatch keys:")

for key in batch.keys():
    print(f"  {key}")


# ============================================================
# PREPARE BATCH
# ============================================================

neighbor_types = batch["neighbor_types"].to(DEVICE)
node_indices = batch["node_indices"].to(DEVICE)
neighbor_hops = batch["neighbor_hops"].to(DEVICE)
neighbor_times = batch["neighbor_times"].to(DEVICE)
edge_index = batch["edge_index"].to(DEVICE)
batch_vec = batch["batch"].to(DEVICE)
labels = batch["labels"].to(DEVICE)

grouped_tf_dict = {
    "grouped_tfs": batch["grouped_tfs"],
    "grouped_indices": batch["grouped_indices"],
    "flat_batch_idx": batch["flat_batch_idx"],
    "flat_nbr_idx": batch["flat_nbr_idx"],
}

agg_batch_dict = {
    "grouped_agg": batch["grouped_agg"],
    "grouped_indices": batch["grouped_indices"],
    "flat_batch_idx": batch["flat_batch_idx"],
    "flat_nbr_idx": batch["flat_nbr_idx"],
}


# ============================================================
# CHECK AGG DIMENSIONS
# ============================================================

print("\nChecking aggregation dimensions...")

for node_type, agg in batch["grouped_agg"].items():

    type_str = train_data.index_to_node_type[node_type]

    actual_dim = agg.shape[-1]

    expected_dim = model.agg_encoder.agg_dim_dict[type_str]

    print(
        f"  {type_str}: "
        f"{tuple(agg.shape)} "
        f"(actual={actual_dim}, expected={expected_dim})"
    )

    assert actual_dim == expected_dim, (
        f"AGG DIMENSION MISMATCH: "
        f"{type_str}: "
        f"{actual_dim} != {expected_dim}"
    )

    if torch.is_tensor(agg):
        n_nan = torch.isnan(agg).sum().item()
        n_inf = torch.isinf(agg).sum().item()
        if n_nan > 0 or n_inf > 0:
            print(f"    ⚠ raw agg has NaN={n_nan}, Inf={n_inf} — will be sanitized inside NeighborAggEncoder")
        # no assertion here anymore — encoder handles it

print("✓ Aggregation dimensions and values OK")


# ============================================================
# FORWARD + BACKWARD
# ============================================================

print("\nRunning forward pass...")

model.train()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001,
)

optimizer.zero_grad(set_to_none=True)

pred = model(
    neighbor_types,
    node_indices,
    neighbor_hops,
    neighbor_times,
    grouped_tf_dict,
    agg_batch_dict,
    edge_index=edge_index,
    batch=batch_vec,
)

print(f"Prediction shape: {tuple(pred.shape)}")

assert torch.isfinite(pred).all(), (
    "Prediction contains NaN or Inf!"
)

print("✓ Forward pass OK")
print("✓ Prediction is finite")

# --- NEW: directly verify the agg encoder's own output is clean ---
with torch.no_grad():
    test_agg_out = model.agg_encoder(agg_batch_dict, neighbor_types)
    assert torch.isfinite(test_agg_out).all(), "agg_encoder output contains NaN/Inf despite nan_to_num!"
    print(f"✓ agg_encoder output finite, shape={tuple(test_agg_out.shape)}")


# ============================================================
# LOSS
# ============================================================

pred_for_loss = (
    pred.view(-1)
    if pred.size(1) == 1
    else pred
)

loss = loss_fn(
    pred_for_loss.float(),
    labels,
)

print(f"\nLoss: {loss.item()}")

assert torch.isfinite(loss), (
    "Loss is NaN or Inf!"
)

print("✓ Loss is finite")


# ============================================================
# BACKWARD
# ============================================================

print("\nRunning backward...")

loss.backward()

print("✓ Backward OK")


# ============================================================
# CHECK AGG ENCODER GRADIENTS
# ============================================================

print("\nChecking agg_encoder gradients...")

found = 0
with_grad = 0

for name, param in model.named_parameters():

    if "agg_encoder" not in name:
        continue

    found += 1

    if not param.requires_grad:
        raise RuntimeError(
            f"agg_encoder parameter has "
            f"requires_grad=False: {name}"
        )

    if param.grad is not None:
        with_grad += 1

    print(
        f"  {name}: "
        f"requires_grad={param.requires_grad}, "
        f"gradient={'YES' if param.grad is not None else 'NO'}"
    )


assert found > 0, (
    "No agg_encoder parameters found!"
)

assert with_grad > 0, (
    "agg_encoder parameters received no gradients!"
)


# ============================================================
# SUCCESS
# ============================================================

print("\n")
print("=" * 70)
print("DRY RUN PASSED ✓")
print("=" * 70)
print(f"Batch size       : {BATCH_SIZE}")
print(f"Prediction shape : {tuple(pred.shape)}")
print(f"Loss             : {loss.item():.6f}")
print(f"Agg parameters   : {found}")
print(f"Agg with gradient: {with_grad}")
print("=" * 70)
print("Safe to proceed with DDP training.")
print("=" * 70)