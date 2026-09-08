import featuretools as ft
import pandas as pd
import numpy as np
import h5py
import torch


def build_entityset(dataset, db):
    es = ft.EntitySet(id=dataset)

    for table_name, table in db.table_dict.items():
        es.add_dataframe(
            dataframe_name=table_name,
            dataframe=table.df,
            index=table.pkey_col,
            time_index=table.time_col if table.time_col is not None else None,
        )

    for child_table_name, child_table in db.table_dict.items():
        for fk_col, parent_table_name in child_table.fkey_col_to_pkey_table.items():
            parent_table = db.table_dict[parent_table_name]
            es.add_relationship(
                parent_dataframe_name=parent_table_name,
                parent_column_name=parent_table.pkey_col,
                child_dataframe_name=child_table_name,
                child_column_name=fk_col,
            )
    return es


def table_needs_cutoff(es, target_table, max_depth):
    # crude but effective: any dataframe within max_depth hops that has a time_index
    seen, frontier = {target_table}, {target_table}
    for _ in range(max_depth):
        nxt = set()
        for rel in es.relationships:
            if rel.parent_dataframe.ww.name in frontier:
                nxt.add(rel.child_dataframe.ww.name)
            if rel.child_dataframe.ww.name in frontier:
                nxt.add(rel.parent_dataframe.ww.name)
        frontier = nxt - seen
        seen |= nxt
    return any(es[t].ww.time_index is not None for t in seen)


def precompute_agg_features(dataset, db, max_depth=2, agg_primitives=None, out_path="agg_features.h5"):
    agg_primitives = agg_primitives or ["count", "sum", "mean", "max", "min"]
    es = build_entityset(dataset, db)

    with h5py.File(out_path, "w") as hf:
        for table_name, table in db.table_dict.items():
            if table_needs_cutoff(es, table_name, max_depth):
                cutoff_time_df = pd.DataFrame({
                    "instance_id": table.df[table.pkey_col],
                    "time": pd.to_datetime(table.df[table.time_col]) if table.time_col
                            else pd.Timestamp.now(),  # see note below
                })
            else:
                cutoff_time_df = None

            fm, feature_defs = ft.dfs(
                entityset=es,
                target_dataframe_name=table_name,
                agg_primitives=agg_primitives,
                max_depth=max_depth,
                cutoff_time=cutoff_time_df,
            )

            grp = hf.create_group(table_name)
            grp.create_dataset("matrix", data=fm.values.astype(np.float32))
            grp.create_dataset("feature_names", data=np.array([str(f) for f in feature_defs], dtype="S"))

    return out_path


import re

PRIMITIVE_VOCAB = {"COUNT": 0, "SUM": 1, "MEAN": 2, "MAX": 3, "MIN": 4}

def parse_feature_names(raw_names):
    primitive_ids, hop_ids, col_ids, col_vocab = [], [], [], {}
    for raw in raw_names:
        name = raw.decode() if isinstance(raw, bytes) else raw
        m = re.match(r"(\w+)\((.+)\)", name)
        if m is None:
            primitive_ids.append(0); hop_ids.append(0)
            col = name
        else:
            prim, inner = m.groups()
            primitive_ids.append(PRIMITIVE_VOCAB.get(prim.upper(), 0))
            hop_ids.append(inner.count("."))   # rough depth proxy — refine if needed
            col = inner
        col_vocab.setdefault(col, len(col_vocab))
        col_ids.append(col_vocab[col])
    return {
        "primitive_ids": torch.tensor(primitive_ids, dtype=torch.long),
        "hop_ids": torch.tensor(hop_ids, dtype=torch.long),
        "col_ids": torch.tensor(col_ids, dtype=torch.long),
    }