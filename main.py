import argparse
import sys
from pathlib import Path
from typing import Mapping, Optional, Callable

import dgl
import joblib
import numpy as np
import torch
from dgl import load_graphs, remove_self_loop
from torch.hub import tqdm
import json
import pandas as pd

##################### Nirvana ##########################################
from nirvana_utils import copy_out_to_snapshot, copy_snapshot_to_out  ###
from utils import (
    Config,
    FEATURES_DATA_NAME,
    LABELS_DATA_NAME,
    TRAIN_MASK_DATA_NAME,
    VAL_MASK_DATA_NAME,
    TEST_MASK_DATA_NAME,
    OUTPUT_MASK_NAME,
    NODE_ID_DATA_NAME,
    construct_subgraph_from_blocks,
    init_dataloader,
    write_output_to_YT,
    get_config,
    prepare_json_input
)

from models.gnn_initial_and_plre import create_graph_model


##################### Nirvana ##########################################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.append("./")

# TODO HERE
class TrainEval:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        criterion,
        device,
        batch_size,
        num_epochs,
        mode,
        test_dataloader=None,
        val_every_steps: int = 5,
        early_stopping_steps: int = 40,
        state_dict_file: Optional[str] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epoch = num_epochs
        self.device = device
        self.batch_size = batch_size
        self.val_every_steps = val_every_steps
        self.early_stopping_steps = early_stopping_steps

        self.node_data_names = [FEATURES_DATA_NAME, TRAIN_MASK_DATA_NAME, VAL_MASK_DATA_NAME, TEST_MASK_DATA_NAME, LABELS_DATA_NAME, NODE_ID_DATA_NAME]

        self.mode = mode

        if state_dict_file is not None:
            state_dict: Mapping[str, torch.FloatTensor] = torch.load(state_dict_file, map_location=device)
            self.model.load_state_dict(state_dict)
            print("State dict is loaded")

    def get_logits_and_labels_for_output_nodes(self, subgraph: dgl.DGLGraph, mask_data_name: torch.Tensor):
        output_nodes_mask = subgraph.ndata[OUTPUT_MASK_NAME]
        input_features = subgraph.ndata[FEATURES_DATA_NAME]
        all_output_mask = subgraph.ndata[mask_data_name]

        all_logits = self.model(subgraph, input_features)
        
        output_nodes_train_mask = all_output_mask[output_nodes_mask]
        
        output_nodes_logits = all_logits[output_nodes_mask]
        output_nodes_labels = subgraph.ndata[LABELS_DATA_NAME][output_nodes_mask]
        output_nodes_ids = subgraph.ndata[NODE_ID_DATA_NAME][output_nodes_mask]

        # if apply_train_val_mask:
        #     logits = output_nodes_logits[output_nodes_train_mask]
        #     labels = output_nodes_labels[output_nodes_train_mask]
        #     ids = output_nodes_ids[output_nodes_train_mask]

        # else:
        #     logits = output_nodes_logits
        #     labels = output_nodes_labels
        #     ids = output_nodes_ids
        
        logits = output_nodes_logits[output_nodes_train_mask]
        labels = output_nodes_labels[output_nodes_train_mask]
        ids = output_nodes_ids[output_nodes_train_mask]

        return dict(
            output_nodes_train_val_mask=output_nodes_train_mask,
            logits=logits,
            labels=labels,
            ids=ids,
        )

    def get_subgraph_from_data(self, data) -> dgl.DGLGraph:
        _, _, layers_subgraphs = data

        subgraph: dgl.DGLGraph = construct_subgraph_from_blocks(
            blocks=layers_subgraphs,
            node_attributes_to_copy=self.node_data_names,
            batch_size=self.batch_size,
            device=self.device,
        )

        return subgraph

    def train_fn(self, current_epoch):
        self.model.train()
        total_loss = 0.0
        tk = tqdm(self.train_dataloader, desc="EPOCH" + "[TRAIN]" + str(current_epoch) + "/" + str(self.epoch))

        for t, data in enumerate(tk, 1):
            subgraph: dgl.DGLGraph = self.get_subgraph_from_data(data)

            self.optimizer.zero_grad()

            return_dict = self.get_logits_and_labels_for_output_nodes(subgraph, mask_data_name=TRAIN_MASK_DATA_NAME)
            loss = self.criterion(return_dict["logits"], return_dict["labels"])

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            tk.set_postfix({"Loss": "%6f" % float(total_loss / t)})

        return total_loss / len(self.train_dataloader)

    @torch.no_grad()
    def eval_fn(self, current_epoch):
        self.model.eval()
        total_loss = 0.0
        tk = tqdm(self.val_dataloader, desc="EPOCH" + "[VALID]" + str(current_epoch) + "/" + str(self.epoch))

        for t, data in enumerate(tk, 1):
            subgraph: dgl.DGLGraph = self.get_subgraph_from_data(data)

            return_dict = self.get_logits_and_labels_for_output_nodes(subgraph, mask_data_name=VAL_MASK_DATA_NAME)
            loss = self.criterion(return_dict["logits"], return_dict["labels"])

            total_loss += loss.item()
            tk.set_postfix({"Loss": "%6f" % float(total_loss / t)})

        return total_loss / len(self.val_dataloader)

    @torch.no_grad()
    def test(self) -> tuple[dict[int, float], dict[int, float]]:
        self.model.eval()

        def list_of_tensors_to_numpy_flat(array, apply_func: Optional[Callable[[torch.Tensor], torch.Tensor]]=None):
            plain_array = torch.cat(array, dim=0).cpu().reshape(-1)
            
            if apply_func is not None:
                plain_array = apply_func(plain_array)
            return plain_array.numpy()

        predictions: list[torch.Tensor] = []  # type: ignore
        labels: list[torch.Tensor] = []  # type: ignore
        output_nodes_ids: list[torch.Tensor] = []  # type: ignore

        tk = tqdm(self.test_dataloader, desc="TEST")

        total_loss = 0.0

        for t, data in enumerate(tk, 1):
            subgraph: dgl.DGLGraph = self.get_subgraph_from_data(data)

            return_dict = self.get_logits_and_labels_for_output_nodes(subgraph, mask_data_name=TEST_MASK_DATA_NAME)

            logits = return_dict["logits"]
            true_labels = return_dict["labels"]
            output_batch_ids = return_dict["ids"]


            loss = self.criterion(logits, true_labels)

            total_loss += loss.item()

            labels.append(true_labels.cpu())
            
            output_nodes_ids.append(output_batch_ids.cpu())

            
            predictions.append(logits.cpu())

            tk.set_postfix({"Loss": "%6f" % float(total_loss / t)})

        predictions: np.ndarray = list_of_tensors_to_numpy_flat(predictions, apply_func=torch.sigmoid)
        labels: np.ndarray = list_of_tensors_to_numpy_flat(labels)
        output_nodes_ids: np.ndarray = list_of_tensors_to_numpy_flat(output_nodes_ids)

        id2logits_df = pd.DataFrame(
            data={NODE_ID_DATA_NAME: output_nodes_ids, "score": predictions}, columns=[NODE_ID_DATA_NAME, "score"]
        )

        return id2logits_df

    def train_and_test(self):
        if self.mode == "training":
            best_valid_loss = np.inf
            best_train_loss = np.inf

            for i in range(1, self.epoch + 1):
                train_loss = self.train_fn(i)

                if i % self.val_every_steps == 0:
                    val_loss = self.eval_fn(i)

                    if val_loss < best_valid_loss:
                        torch.save(self.model.state_dict(), "checkpoints/best-weights.pt")
                        print("Saved Best Weights")
                        best_valid_loss = val_loss
                        best_train_loss = train_loss

                        #############
                        # IMPORTANT #
                        #############
                        copy_out_to_snapshot("./")

                torch.cuda.empty_cache()

            print(f"Training Loss : {best_train_loss}")
            print(f"Valid Loss : {best_valid_loss}")
        else:
            print(f"The mode is {self.mode}, going straight to testing")
            
            
        torch.save(self.model.state_dict(), "checkpoints/last-weights.pt")
        print("Saved Last Weights")
        copy_out_to_snapshot("./")

        if self.test_dataloader is not None:
            print("Performing test on test dataloader")

            id2logits = self.test()

            return id2logits


        return {}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Graph Neural Network for fraud prediction")
    
    parser.add_argument("--datadir", type=Path, help="Directory with data", default="./data")

    parser.add_argument("--mode", choices=["training", "inference"], default="training")
    
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Device index for GPU"
    )
    
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    return parser




def main():
    args = get_parser().parse_args()
    
    if args.device is not None: # for local launch with manual device choise
        DEVICE = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device is ", DEVICE)
    
    mode: str = args.mode
    debug_mode: bool = args.debug
    datadir: Path = args.datadir

    #############
    # IMPORTANT #
    #############
    copy_snapshot_to_out("./")


    # NOTE: this is not the final implementation of training and evaluation!
    if mode == "training":
        weights_file = None
        train_metadata_file = None
        
        print(f"The mode is {mode}, launching initial training...")
    else:
        weights_file = "checkpoints/last-weights.pt"
        train_metadata_file = "checkpoints/train_metadata"

        print(f"The mode is {mode}, picking preempted weights...")


    config: Config = get_config(debug_mode=debug_mode)
    MODEL_PARAMS = config.MODEL_PARAMS
    TRAINING_PARAMETERS = config.TRAINING_PARAMS
    table_output_root_path: str = config.table_output_root_path


    graph, scaler, train_metadata, node_index_to_id_mapper = prepare_json_input(data_dir=datadir, train_metadata_file=train_metadata_file)
    
    joblib.dump(scaler, "checkpoints/scaler.bin")
    with open("checkpoints/train_metadata", "wb") as write_handler:
        joblib.dump(train_metadata, write_handler)
    print("Successfully created graphs and dumped metadata")
        

    if config.remove_self_loops:
        graph = remove_self_loop(graph)
    
    num_input_features = graph.ndata[FEATURES_DATA_NAME].shape[1]
    
    print("Successfully created graph and removed self-loops if necessary")        
        

    MODEL_PARAMS.update(dict(num_input_features=num_input_features))
    model = create_graph_model(model_name=config.model_type, model_params=MODEL_PARAMS).to(DEVICE)

    loss_func = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAINING_PARAMETERS["learning_rate"],
        weight_decay=TRAINING_PARAMETERS["weight_decay"],
    )

    sampler = dgl.dataloading.NeighborSampler(
        fanouts=[TRAINING_PARAMETERS["max_num_neighbors"]] * MODEL_PARAMS["num_encoder_layers"]
    )

    batch_size = TRAINING_PARAMETERS["batch_size"]
    num_workers = TRAINING_PARAMETERS["num_workers"]

    train_loader = init_dataloader(graph, sampler, DEVICE, batch_size=batch_size, num_workers=num_workers)
    val_loader = init_dataloader(graph, sampler, DEVICE, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    test_loader = init_dataloader(graph, sampler, DEVICE, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    trainer = TrainEval(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        criterion=loss_func,
        device=DEVICE,
        batch_size=batch_size,
        mode=mode,
        num_epochs=TRAINING_PARAMETERS["num_epochs"],
        val_every_steps=TRAINING_PARAMETERS["val_every_steps"],
        state_dict_file=weights_file,
    )
    
    print("Initialized trainer")
    index2logits_df: Optional[pd.DataFrame] = trainer.train_and_test()
    index2logits_df[NODE_ID_DATA_NAME] = index2logits_df[NODE_ID_DATA_NAME].map(node_index_to_id_mapper)
    if mode == "inference" and not debug_mode:
        
        print(f"Predictions:\n{index2logits_df}")
        
        index2logits_df.to_csv("index2logits_df.csv")
        index2logits_list_of_dicts = index2logits_df.to_dict('records')

        mr_table_output: dict[str, str] = write_output_to_YT(output=index2logits_list_of_dicts, 
                                                            table_path_root=table_output_root_path)
                
        with open("MR_TABLE", "w") as out_handler:
            json.dump(mr_table_output, out_handler)
    else:
        if mode == "training" and not debug_mode:
            print("Mode is `training`, there is no data to load")
        else:
            print("Debug mode is activated, skipping uploading to YT")
            
    copy_out_to_snapshot("./", dump=True)

if __name__ == "__main__":
    main()
