import os
import random
import string
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Optional, Dict, ClassVar

from dataclasses import dataclass
import yaml
import dgl
import joblib
import numpy as np
import torch
import ujson
import yt.wrapper as yt
from sklearn.preprocessing import StandardScaler

sys.path.append("./")

from data_preparation import main_prepare_mr_tables

OUTPUT_MASK_NAME = "output_mask"
FEATURES_DATA_NAME = "features"
MASK_DATA_NAME = "mask"
LABELS_DATA_NAME = "labels"
USERID_DATA_NAME = "userid"

# MODE_FIELD = "mode"

YT_TOKEN = os.environ.get("YT_TOKEN")


@dataclass
class Config:
    # Data options
    remove_self_loops: bool = True
    table_output_root_path: str = "//tmp/"
    model_type: str = "GNN"
    
    # Training parameters
    batch_size: int = 2000000
    num_epochs: int = 75
    max_num_neighbors: int = -1 # -1 for all neighbors to be sampled
    
    num_workers: int = 12
    learning_rate: float = 0.0003
    weight_decay: float = 0.00001
    
    val_every_steps: int = 25
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
    lite: bool = True # lite Linear block option
    

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
        

def get_config(config_dir: Path):
    yaml_config = config_dir / "CONFIG.yaml"
    
    if not yaml_config.exists():
        raise f"No configs were found in {str(config_dir)}. Supported names are 'config.json' and 'config.yaml'"

    with open(yaml_config) as f_read:
        config_dict: Dict[str, Any] = yaml.safe_load(f_read)
        
    config = Config(**config_dict)
    
    return config    


def _scale_features(train_val_test_features_container: List[List[float]], scaler_state_file: Path = None):
    if scaler_state_file.exists():
        import joblib

        scaler = joblib.load(filename=scaler_state_file)
    else:
        scaler = StandardScaler()
        scaler.fit(train_val_test_features_container[0])  # first is train

    transformed_container = [scaler.transform(features) for features in train_val_test_features_container]

    return transformed_container, scaler


def _construct_dgl_graph(
    adjacency_matrix_rows_cols, features: np.ndarray, targets: np.ndarray, mask: np.ndarray, user_ids: np.ndarray
):
    row_coordinates, col_coordinates = (
        adjacency_matrix_rows_cols["row_coords"],
        adjacency_matrix_rows_cols["col_coords"],
    )

    row_coordinates = torch.tensor(row_coordinates).long()
    col_coordinates = torch.tensor(col_coordinates).long()
    graph = dgl.graph(data=(row_coordinates, col_coordinates), idtype=torch.int32)
    graph.ndata[FEATURES_DATA_NAME] = torch.tensor(features, dtype=torch.float32)
    graph.ndata[LABELS_DATA_NAME] = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
    graph.ndata[MASK_DATA_NAME] = torch.tensor(mask, dtype=torch.bool).reshape(-1, 1)

    graph.ndata[USERID_DATA_NAME] = torch.tensor(user_ids, dtype=torch.long).reshape(-1, 1)

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
    scaler_state_filename: Path = data_dir / "scaler.bin"
    json_input_filename: Path = data_dir / "JSON_INPUT.json"

    with open(json_input_filename) as handler:
        mr_tables = ujson.load(handler)["mr_tables"]

    if train_metadata_file is not None:
        train_metadata = joblib.load(open(train_metadata_file, "rb"))
    else:
        train_metadata = None

    input_dict = main_prepare_mr_tables(mr_tables=mr_tables, token=YT_TOKEN, train_metadata=train_metadata)
    
    
    masks_dict: dict[str, np.ndarray] = input_dict["masks"]

    test_data_dict: dict[str, np.ndarray] = input_dict["test_data"]
    train_data_dict: Optional[dict[str, np.ndarray]] = input_dict.get("train_data", test_data_dict)  # TODO make more flexible


    test_mask = masks_dict["test_mask"]
    train_mask = masks_dict.get("train_mask", test_mask)
    val_mask = masks_dict.get("val_mask", test_mask)
    
    train_features = train_data_dict[FEATURES_DATA_NAME]
    val_features = train_data_dict[FEATURES_DATA_NAME]
    test_features = test_data_dict[FEATURES_DATA_NAME]

    [train_features, val_features, test_features], scaler = _scale_features(
        train_val_test_features_container=[train_features, val_features, test_features],
        scaler_state_file=scaler_state_filename,
    )

    train_targets = val_targets = train_data_dict["targets"]
    test_targets = test_data_dict["targets"]

    train_adjacency = val_adjacency = train_data_dict["adjacency_matrix_tools"]
    test_adjacency = test_data_dict["adjacency_matrix_tools"]

    train_user_ids = val_user_ids = train_data_dict["user_ids"]
    test_user_ids = test_data_dict["user_ids"]

    _zip_container = [
        [train_adjacency, train_features, train_targets, train_mask, train_user_ids],
        [val_adjacency, val_features, val_targets, val_mask, val_user_ids],
        [test_adjacency, test_features, test_targets, test_mask, test_user_ids],
    ]

    graph_train, graph_valid, graph_test = (
        _construct_dgl_graph(*adj_feats_targets_mask) for adj_feats_targets_mask in _zip_container
    )

    return [graph_train, graph_valid, graph_test], scaler, input_dict["train_metadata"]


def write_output_to_YT(output: list[dict[str, Any]], table_path_root: str = "//home/yr/fvelikon/tmp") -> dict[str, str]:
    @yt.yt_dataclass
    class Row:
        userid: str
        score: float

    client = yt.YtClient(proxy="hahn", token=YT_TOKEN)

    random_name = "".join(random.choices(string.ascii_lowercase, k=20))
    table_path = "{}/table_antifraud_{}".format(table_path_root, random_name)

    table_rows: List[Row] = [Row(userid=row["userid"], score=row["score"]) for row in output]

    print(f"Trying to save the table to {table_path}")

    yt.write_table_structured(table_path, Row, table_rows, client=client)

    print(f"Table was saved to {table_path}")

    mr_table = dict(cluster="hahn", table=table_path)

    return mr_table
