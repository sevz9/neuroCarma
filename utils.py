import os
import random
import string
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Optional, Dict, ClassVar
import gc

import dgl.graphbolt as gb
import glob
from pydantic import validate_arguments
from dataclasses import dataclass
import yaml
import dgl
import joblib
import numpy as np
import torch
import json
import yt.wrapper as yt
from sklearn.preprocessing import StandardScaler
sys.path.append("./")

from data_preparation import main_prepare_mr_tables
from nirvana_utils import copy_out_to_snapshot

OUTPUT_MASK_NAME = "output_mask"
FEATURES_DATA_NAME = "features"

TRAIN_MASK_DATA_NAME = "train_mask"
VAL_MASK_DATA_NAME = "val_mask"
TEST_MASK_DATA_NAME = "test_mask"

LABELS_DATA_NAME = "target"
NODE_ID_DATA_NAME = "key"

MASK_DATA_NAME = "mask"


YT_TOKEN = os.environ.get("YT_TOKEN")

@validate_arguments
@dataclass
class Config:
    # Data options
    remove_self_loops: bool = True
    table_output_root_path: str = "//tmp/"
    model_type: str = "GNN"

    # Training parameters
    batch_size: int = 2000000
    num_epochs: int = 2
    max_num_neighbors: int = -1  # -1 for all neighbors to be sampled

    num_workers: int = 1
    learning_rate: float = 0.0003
    weight_decay: float = 0.00001

    val_every_steps: int = 2
    early_stopping_steps: int = 1000

    # Model Parameters
    num_hidden_features: int = 128
    normalisation_name: str = "batch"

    # Convolutiom parameters
    convolution_name: str = "sage"
    convolution_params: ClassVar[Dict[str, str]] = {
        "aggregator_type": "mean",
    }

    activation_name: str = "gelu"
    apply_skip_connection: bool = True
    num_preprocessing_layers: int = 1
    num_encoder_layers: int = 2

    num_predictor_layers: int = 1

    # PLRE parameters
    n_frequencies: int = 48
    frequency_scale: float = 0.02
    d_embedding: int = 16
    lite: bool = True  # lite Linear block option

    @property
    def MODEL_PARAMS(self):
        return dict(
            num_hidden_features=self.num_hidden_features,
            normalisation_name=self.normalisation_name,
            convolution_name=self.convolution_name,
            convolution_params=self.convolution_params,
            activation_name=self.activation_name,
            apply_skip_connection=self.apply_skip_connection,
            num_preprocessing_layers=self.num_preprocessing_layers,
            num_encoder_layers=self.num_encoder_layers,
            num_predictor_layers=self.num_predictor_layers,
            n_frequencies=self.n_frequencies,
            frequency_scale=self.frequency_scale,
            d_embedding=self.d_embedding,
        )

    @property
    def TRAINING_PARAMS(self):
        return dict(
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            max_num_neighbors=self.max_num_neighbors,
            num_workers=self.num_workers,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            val_every_steps=self.val_every_steps,
            early_stopping_steps=self.early_stopping_steps,
        )

    def to_dict(self) -> Dict[str, Any]:
        def is_param(param_name: str):
            return all(
                (
                    not param_name.startswith("_"),
                    param_name not in {"to_dict", "TRAINING_PARAMS", "MODEL_PARAMS"},
                    not callable(getattr(self, param_name)),
                )
            )

        params = filter(is_param, dir(self))

        return {param_name: getattr(self, param_name) for param_name in params}



def get_config(config_dir: Path = Path().cwd(), debug_mode=False):
    if debug_mode:
        return Config()  # default options for debugging
    yaml_config = config_dir / "CONFIG.yaml"

    if not yaml_config.exists():
        raise f"No configs were found in {str(config_dir)}. Supported names are 'config.json' and 'config.yaml'"

    with open(yaml_config) as f_read:
        config_dict: Dict[str, Any] = yaml.safe_load(f_read)

    config = Config(**config_dict)

    return config


def _scale_features(features: np.ndarray, scaler_state_file: Path = None):
    if scaler_state_file.exists():
        import joblib

        scaler = joblib.load(filename=scaler_state_file)
    else:
        scaler = StandardScaler()
        scaler.fit(features)  # fits during training phase

    transformed_features = scaler.transform(features)

    return transformed_features, scaler



def create_graphbolt_dataloader(graph, features, train_val_test_set, bath_size, fanouts_list, shuffle, node_feature_keys, device, num_workers):
    datapipe = gb.ItemSampler(train_val_test_set, batch_size=bath_size, shuffle=shuffle)
    datapipe = datapipe.sample_neighbor(graph, fanouts_list)
    
    datapipe = datapipe.fetch_feature(features, node_feature_keys=node_feature_keys)
    datapipe = datapipe.copy_to(device)
    dataloader = gb.DataLoader(datapipe, num_workers=num_workers)
    
    return dataloader


def _construct_dgl_graph(
    edges: np.ndarray, features: np.ndarray, targets: np.ndarray, mask: np.ndarray
):
    row_coordinates, col_coordinates = edges[0, :], edges[1, :]

    row_coordinates = torch.tensor(row_coordinates).long()
    col_coordinates = torch.tensor(col_coordinates).long()
    graph = dgl.graph(data=(row_coordinates, col_coordinates), idtype=torch.int32, num_nodes=features.shape[0])
    graph.ndata[FEATURES_DATA_NAME] = torch.tensor(features, dtype=torch.float32)
    graph.ndata[LABELS_DATA_NAME] = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
    graph.ndata[MASK_DATA_NAME] = torch.tensor(mask, dtype=torch.bool).reshape(-1, 1)
# NODE_ID_DATA_NAME
#     graph.ndata[MASK_DATA_NAME] = torch.tensor(mask, dtype=torch.bool).reshape(-1, 1)

    return graph


def get_features_and_labels_from_a_graph(graph: dgl.DGLGraph):
    mask = graph.ndata[MASK_DATA_NAME].bool()
    features = graph.ndata[FEATURES_DATA_NAME][mask].numpy()
    labels = graph.ndata[LABELS_DATA_NAME][mask].numpy()

    return features, labels


def standard_graph_collate(graph_container):
    graph = graph_container[0]

    features = graph.ndata["features"]
    labels = graph.ndata["labels"]
    mask = graph.ndata["mask"]

    return graph, features, labels, mask


def init_dataloader(
    graph: dgl.DGLGraph,
    sampler: dgl.dataloading.Sampler,
    device: str = "cpu",
    shuffle: bool = True,
    batch_size: int = 10_000,
    num_workers: int = 12,
):
    dataloader = dgl.dataloading.DataLoader(
        graph=graph,
        indices=torch.arange(graph.num_nodes(), dtype=torch.int32),
        device=device,
        graph_sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=batch_size,
        drop_last=False,
        use_prefetch_thread=True,
        pin_prefetcher=True,
    )

    return dataloader


def construct_subgraph_from_blocks(
    blocks: list[Any],
    batch_size: int,
    node_attributes_to_copy: list[str],
    device: str,
) -> dgl.DGLGraph:
    """
    Constructs a copy of a Message flow graphs (MFG), defined as a list of MFGs.

    NOTE: this function is an example of constructing graph for node classification tasks, graph obly contains node features


    params:

    `blocks`: list of consecutive message flow graphs, len(blocks) == number of layers in graph convolution
    `batch_size`: number of destination nodes
    `node_attributes_to_copy`: list of names of node attributes to copy to a new graph
    """

    merged_block = deepcopy(dgl.merge([dgl.block_to_graph(b) for b in blocks]))
    # merged_block = dgl.merge([dgl.block_to_graph(b) for b in blocks])

    row_coords, col_coords = merged_block.edges()

    number_of_nodes = merged_block.srcdata[FEATURES_DATA_NAME].shape[0]
    
    new_graph = dgl.graph(data=(row_coords, col_coords), num_nodes=number_of_nodes)
    

    for node_data_name in node_attributes_to_copy:

        try:
            new_graph.ndata[node_data_name] = merged_block.srcdata[node_data_name]
        except dgl._ffi.base.DGLError as e:
            print(f"{node_data_name=} {merged_block.srcdata[node_data_name].shape=} {new_graph.num_nodes()=} {merged_block.num_nodes()=}")
            raise e

    # create mask marking only destination nodes, which are needed for
    num_of_nodes = new_graph.num_nodes()
    output_mask = torch.zeros(num_of_nodes).bool()
    output_mask[:batch_size] = True
    new_graph.ndata[OUTPUT_MASK_NAME] = output_mask.to(device)

    del merged_block

    return new_graph.to(device)


def prepare_json_input(data_dir: Path, train_metadata_file: Optional[str] = None):

    dataset_base_dir = "checkpoints/dataset/"
    os.makedirs(dataset_base_dir, exist_ok=True)

    scaler_state_filename: Path = data_dir / "scaler.bin"
    json_input_filename: Path = data_dir / "JSON_INPUT.json"

    with open(json_input_filename) as handler:
        input_json = json.load(handler)
        features_mr_table = input_json["features_mr_table"]
        edges_mr_table = input_json["edges_mr_table"]

    if train_metadata_file is not None:
        train_metadata = joblib.load(open(train_metadata_file, "rb"))
    else:
        train_metadata = None
        
        
    # prepare for graceful restart of the download:
    _loading_metadata_path = "./checkpoints/load_metadata.json"
    if os.path.exists(_loading_metadata_path):
        with open(_loading_metadata_path) as handler:
            loading_metadata = json.load(handler)
            
            features_table_loaded = loading_metadata["features_table_loaded"]
            edge_index_rows_loaded = loading_metadata["edge_index_rows_loaded"]            
    else:
        features_table_loaded = False
        edge_index_rows_loaded = 0
    
    print(f"Optional graceful restart is available: {features_table_loaded=}, edge_index_rows_loaded={edge_index_rows_loaded/1e6}M")
    
    input_dict = main_prepare_mr_tables(
        features_mr_table=features_mr_table,
        edges_mr_table=edges_mr_table,
        token=YT_TOKEN,
        train_metadata=train_metadata,
        
        features_table_loaded=features_table_loaded,
        edge_index_rows_loaded=edge_index_rows_loaded,
        _loading_metadata_path=_loading_metadata_path
    )

    masks_dict: dict[str, np.ndarray] = input_dict["masks"]

    edges = np.load(input_dict["edges_file"])


    targets = input_dict["targets"]
    
    test_mask = masks_dict["test_mask"].astype(bool)
    train_mask = masks_dict["train_mask"].astype(bool)
    val_mask = masks_dict["val_mask"].astype(bool)

    features = input_dict[FEATURES_DATA_NAME].astype(np.float32)
    features, scaler = _scale_features(
        features=features,
        scaler_state_file=scaler_state_filename,
    )
    num_features = features.shape[1]


    graph_train, graph_val, graph_test = [_construct_dgl_graph(edges=edges, features=features, targets=targets, mask=mask) for mask in [train_mask, val_mask, test_mask]]

    return [graph_train, graph_val, graph_test], scaler, input_dict["train_metadata"], input_dict["node_index_to_id_mapper"]





def prepare_json_input_graphbolt(data_dir: Path, train_metadata_file: Optional[str] = None):
    
    def create_dataset_and_return():
        dataset = gb.OnDiskDataset(dataset_base_dir, auto_cast_to_optimal_dtype=False).load()
        graph = dataset.graph
        print(f"Loaded graph: {graph}\n")
        feature = dataset.feature
        print(f"Loaded feature store: {feature}\n")

        tasks = dataset.tasks
        nc_task = tasks[0]
        print(f"Loaded node classification task: {nc_task}\n")
        
        # # delete_all_unnecessary files: <--- DGL Can't handle it for now. skipping
        # for npy_file in glob.glob(f"{dataset_base_dir}/*.npy"):
        #     os.remove(npy_file)
        #     print(f"Deleted {npy_file} as its ancestor is stored in preprocessed directory")
            
        #     filename = os.path.basename(npy_file)
        #     if filename == "edges.npy":
        #         continue
        #     preprocessed_filepath = os.path.join(dataset_base_dir, "preprocessed", filename)
        #     os.symlink(src=preprocessed_filepath, dst=npy_file)
        #     print("Created symlink to the preprocessed file")

        
        copy_out_to_snapshot("./")
        return dataset, num_features, scaler, input_dict["train_metadata"], node_index_to_id_mapper

    # save all files in a special directory and thus preprocess graph
    dataset_base_dir = "checkpoints/dataset/"
    os.makedirs(dataset_base_dir, exist_ok=True)


    scaler_state_filename: Path = data_dir / "scaler.bin"
    json_input_filename: Path = data_dir / "JSON_INPUT.json"

    with open(json_input_filename) as handler:
        input_json = json.load(handler)
        features_mr_table = input_json["features_mr_table"]
        edges_mr_table = input_json["edges_mr_table"]

    if train_metadata_file is not None:
        train_metadata = joblib.load(open(train_metadata_file, "rb"))
    else:
        train_metadata = None
        
        
    # prepare for graceful restart of the download:
    _loading_metadata_path = "./checkpoints/load_metadata.json"
    if os.path.exists(_loading_metadata_path):
        with open(_loading_metadata_path) as handler:
            loading_metadata = json.load(handler)
            
            features_table_loaded = loading_metadata["features_table_loaded"]
            edge_index_rows_loaded = loading_metadata["edge_index_rows_loaded"]            
    else:
        features_table_loaded = False
        edge_index_rows_loaded = 0
    
    print(f"Optional graceful restart is available: {features_table_loaded=}, edge_index_rows_loaded={edge_index_rows_loaded/1e6}M")
    
    input_dict = main_prepare_mr_tables(
        features_mr_table=features_mr_table,
        edges_mr_table=edges_mr_table,
        token=YT_TOKEN,
        train_metadata=train_metadata,
        
        features_table_loaded=features_table_loaded,
        edge_index_rows_loaded=edge_index_rows_loaded,
        _loading_metadata_path=_loading_metadata_path
    )

    masks_dict: dict[str, np.ndarray] = input_dict["masks"]



    test_mask = masks_dict["test_mask"].astype(bool)
    train_mask = masks_dict["train_mask"].astype(bool)
    val_mask = masks_dict["val_mask"].astype(bool)

    features = input_dict[FEATURES_DATA_NAME].astype(np.float32)
    features, scaler = _scale_features(
        features=features,
        scaler_state_file=scaler_state_filename,
    )
    num_features = features.shape[1]


    targets = input_dict["targets"].astype(np.float32)
    edges_file = input_dict["edges_file"]
    node_indices = input_dict["node_indices"]
    node_ids = input_dict["node_ids"]
    node_index_to_id_mapper = input_dict["node_index_to_id_mapper"]
    
    
    
    # convert pandas edges table (which is very convenient though) to numpy array as pandas reader reads them in int32 format.
    # import pandas as pd
    # edges_numpy = pd.read_csv(edges_file, dtype=np.int64).values.T
    gc.collect()
    # breakpoint()
    # edges_path_new = edges_file # os.path.join(dataset_base_dir, "edges.npy")
    metadata_path = os.path.join(dataset_base_dir, "metadata.yaml")

    if os.path.exists(metadata_path):
        return create_dataset_and_return()
    
    
    features_path = os.path.join(dataset_base_dir, "features.npy")
    node_indices_path = os.path.join(dataset_base_dir, "node_ids.npy")
    
    train_node_indices_path = os.path.join(dataset_base_dir, "train_node_indices.npy")
    val_node_indices_path = os.path.join(dataset_base_dir, "val_node_indices.npy")
    test_node_indices_path = os.path.join(dataset_base_dir, "test_node_indices.npy")

    train_labels_path = os.path.join(dataset_base_dir, "train_labels.npy")
    val_labels_path = os.path.join(dataset_base_dir, "val_labels.npy")
    test_labels_path = os.path.join(dataset_base_dir, "test_labels.npy")

    
    train_node_indices = node_indices[train_mask].astype(np.int64)
    val_node_indices = node_indices[val_mask].astype(np.int64)
    test_node_indices = node_indices[test_mask].astype(np.int64)
    
    train_labels = targets[train_mask]
    val_labels = targets[val_mask]
    test_labels = targets[test_mask]
    
    del targets
    del node_indices
    
    
    if len(test_node_indices) == 0:
        test_node_indices = val_node_indices
        test_labels = val_labels
    
    for file, filepath in [(features, features_path),
                        #    (node_indices, node_indices_path),
                           (train_node_indices, train_node_indices_path),
                           (val_node_indices, val_node_indices_path),
                           (test_node_indices, test_node_indices_path),
                           (train_labels, train_labels_path),
                           (val_labels, val_labels_path),
                           (test_labels, test_labels_path),
                        #    (edges_numpy, edges_path_new)
                           ]:
        np.save(filepath, file)
        
        del file
        gc.collect()
        
        print(f"Saved part of raw graph data to {filepath}")

    edges_path_new = os.path.join(dataset_base_dir, "edges.npy")
    edges_transposed = np.load(edges_path_new).T
    Path(edges_path_new).unlink(missing_ok=True)
    np.save(edges_path_new, edges_transposed)
    
    del edges_transposed


    yaml_content = f"""
dataset_name: antifraud_graph
graph:
  nodes:
    - num: {features.shape[0]}
  edges:
    - format: numpy
      path: {os.path.basename(edges_path_new)}
feature_data:
  - domain: node
    name: features
    format: numpy
    path: {os.path.basename(features_path)}

tasks:
  - name: node_classification
    num_classes: 2
    train_set:
      - data:
          - name: seed_nodes
            format: numpy
            path: {os.path.basename(train_node_indices_path)}
          - name: labels
            format: numpy
            path: {os.path.basename(train_labels_path)}
    validation_set:
      - data:
          - name: seed_nodes
            format: numpy
            path: {os.path.basename(val_node_indices_path)}
          - name: labels
            format: numpy
            path: {os.path.basename(val_labels_path)}
    test_set:
      - data:
          - name: seed_nodes
            format: numpy
            path: {os.path.basename(test_node_indices_path)}
          - name: labels
            format: numpy
            path: {os.path.basename(test_labels_path)}
"""
    print(yaml_content)

    with open(metadata_path, "w") as f:
        f.write(yaml_content)
        
    return create_dataset_and_return()

def write_output_to_YT(output: list[dict[str, Any]], table_path_root: str = "//home/yr/fvelikon/tmp") -> dict[str, str]:
    @yt.yt_dataclass
    class Row:
        key: str
        score: float

    client = yt.YtClient(proxy="hahn", token=YT_TOKEN)

    _random_name = "".join(random.choices(string.ascii_lowercase, k=20))
    
    table_path = os.path.join(table_path_root, "table_antifraud_{}".format(_random_name))

    table_rows: List[Row] = [Row(key=row["key"], score=row["score"]) for row in output]

    print(f"Trying to save the table to {table_path}")

    yt.write_table_structured(table_path, Row, table_rows, client=client)

    print(f"Table was saved to {table_path}")

    mr_table = dict(cluster="hahn", table=table_path)

    return mr_table

