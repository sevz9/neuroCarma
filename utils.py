import os
import random
import string
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Optional, Dict, ClassVar

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

OUTPUT_MASK_NAME = "output_mask"
FEATURES_DATA_NAME = "features"

TRAIN_MASK_DATA_NAME = "train_mask"
VAL_MASK_DATA_NAME = "val_mask"
TEST_MASK_DATA_NAME = "test_mask"

LABELS_DATA_NAME = "target"
NODE_ID_DATA_NAME = "key"


YT_TOKEN = os.environ.get("YT_TOKEN")

# TODO Convolution parameters proper handling
@validate_arguments
@dataclass
class Config:
    # Data options
    remove_self_loops: bool = False
    table_output_root_path: str = "//tmp/"
    model_type: str = "GNN"

    # Training parameters
    batch_size: int = 2000000
    num_epochs: int = 2
    max_num_neighbors: int = -1  # -1 for all neighbors to be sampled

    num_workers: int = 12
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


def _construct_dgl_graph(
    adjacency_matrix_rows_cols,
    features: np.ndarray,
    targets: np.ndarray,
    node_ids: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
):
    row_coordinates, col_coordinates = (
        adjacency_matrix_rows_cols["row_coords"],
        adjacency_matrix_rows_cols["col_coords"],
    )

    row_coordinates = torch.tensor(row_coordinates).long()
    col_coordinates = torch.tensor(col_coordinates).long()
    
    assert len(row_coordinates) == len(col_coordinates)
    graph = dgl.graph(data=(row_coordinates, col_coordinates), idtype=torch.long, num_nodes=len(node_ids))
    graph = dgl.to_simple(graph, writeback_mapping=False)
    
    graph.ndata[FEATURES_DATA_NAME] = torch.tensor(features, dtype=torch.float32)
    graph.ndata[LABELS_DATA_NAME] = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)

    graph.ndata[TRAIN_MASK_DATA_NAME] = torch.tensor(train_mask, dtype=torch.bool).reshape(-1, 1)
    graph.ndata[VAL_MASK_DATA_NAME] = torch.tensor(val_mask, dtype=torch.bool).reshape(-1, 1)
    graph.ndata[TEST_MASK_DATA_NAME] = torch.tensor(test_mask, dtype=torch.bool).reshape(-1, 1)

    graph.ndata[NODE_ID_DATA_NAME] = torch.tensor(node_ids, dtype=torch.long).reshape(-1, 1)

    print(f"{graph.num_edges()=} {graph.num_nodes()=}")
    
    return graph



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
            print(
                f"{node_data_name=} {merged_block.srcdata[node_data_name].shape=} {new_graph.num_nodes()=} {merged_block.num_nodes()=}"
            )
            raise e

    # create mask marking only destination nodes, which are needed for
    num_of_nodes = new_graph.num_nodes()
    output_mask = torch.zeros(num_of_nodes).bool()
    output_mask[:batch_size] = True
    new_graph.ndata[OUTPUT_MASK_NAME] = output_mask.to(device)

    del merged_block

    return new_graph.to(device)


def prepare_json_input(data_dir: Path, train_metadata_file: Optional[str] = None):
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

    test_mask = masks_dict["test_mask"]
    train_mask = masks_dict["train_mask"]
    val_mask = masks_dict["val_mask"]

    features = input_dict[FEATURES_DATA_NAME]
    features, scaler = _scale_features(
        features=features,
        scaler_state_file=scaler_state_filename,
    )

    targets = input_dict["targets"]
    adjacency = input_dict["adjacency_matrix_rows_cols"]
    node_indices = input_dict["node_indices"]
    
    node_index_to_id_mapper = input_dict["node_index_to_id_mapper"]
    

    graph = _construct_dgl_graph(
        adjacency_matrix_rows_cols=adjacency,
        features=features,
        targets=targets,
        node_ids=node_indices,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    return graph, scaler, input_dict["train_metadata"], node_index_to_id_mapper


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
