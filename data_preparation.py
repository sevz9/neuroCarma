from typing import Any, Optional, Sequence, List, Dict, Mapping

import ytreader
import gc
import numpy as np
import ujson as json
import pandas as pd
import yt.wrapper as yt
import joblib


import time
OptionalColumns = Optional[Sequence[str]]

from nirvana_utils import copy_out_to_snapshot, copy_snapshot_to_out

KEY_COLUMN = "key"
TARGET_COLUMN = "target"
TRAIN_MASK_COLUMN = "train_mask"
VAL_MASK_COLUMN = "val_mask"
TEST_MASK_COLUMN = "test_mask"

SERVICE_COLUMNS_IN_FEATURES_TABLE = {KEY_COLUMN, TARGET_COLUMN, TRAIN_MASK_COLUMN, VAL_MASK_COLUMN, TEST_MASK_COLUMN}

def _check_for_nans(df):
    number_of_nans_per_column = df.isna().sum()
    assert (total_nans := number_of_nans_per_column.sum() == 0), f"""Found {total_nans} nan values. Please make sure to handle them before passing features to the Py3DL. 
                                                                Columns with their respective amount of NaNs: {number_of_nans_per_column}"""

def _read_dataframe_from_yt(mr_table, client: yt.YtClient):
    rows = list(client.read_table(mr_table["table"], format="json", unordered=True, enable_read_parallel=True, raw=True))
    rows = [json.loads(row) for row in rows]
    
    df = pd.DataFrame(rows)
    _check_for_nans(df)
    return df


def read_features_table(mr_table, client: yt.YtClient, feature_columns_presented_in_train: OptionalColumns = None) -> Dict[str, np.ndarray or Dict[str, np.ndarray]]:

    def _process_features_columns(df: pd.DataFrame) -> np.ndarray:
        features_df = df[feature_columns]
        
        if feature_columns_presented_in_train is not None: # inference phase
            columns_unpresented_in_df = list(set(feature_columns_presented_in_train) - set(feature_columns))

            print(
                f"Columns which aren't presented in the dataframe but the model was trained using them: {columns_unpresented_in_df}"
            )

            for col in columns_unpresented_in_df:
                features_df[col] = 0.0
                print(f"Imputed unpresented column {col} with 0.0")
        
            features_df = features_df[feature_columns_presented_in_train].astype(np.float32).values
            
            all_features_names = feature_columns_presented_in_train
        else:
            all_features_names = feature_columns
        
        return features_df, all_features_names

    
    df = _read_dataframe_from_yt(mr_table, client=client)
    
    all_columns = list(df.columns.values)
    feature_columns = list(filter(lambda col: col not in SERVICE_COLUMNS_IN_FEATURES_TABLE, all_columns))
    
    # for the inference mode, target column isn't always presented, thus, we can add it as a placeholder:
    if feature_columns_presented_in_train is None and TARGET_COLUMN not in all_columns:
        df[TARGET_COLUMN] = False

    if len(set(all_columns) & SERVICE_COLUMNS_IN_FEATURES_TABLE) != len(SERVICE_COLUMNS_IN_FEATURES_TABLE):
        raise KeyError(f"Some of the obligatory columns ({SERVICE_COLUMNS_IN_FEATURES_TABLE}) are missing! Missing columns are: {SERVICE_COLUMNS_IN_FEATURES_TABLE - set(all_columns)}")
    
    features_values, all_features_names = _process_features_columns(df)
    
    key_column_values = df[KEY_COLUMN].values
    target_column_values = df[TARGET_COLUMN].values.reshape(-1)
    train_mask_column_values = df[TRAIN_MASK_COLUMN].values.astype(bool)
    val_mask_column_values = df[VAL_MASK_COLUMN].values.astype(bool)
    test_mask_column_values = df[TEST_MASK_COLUMN].values.astype(bool)
    
    
    print(f"{features_values.shape=}, {features_values=}")
    
    return dict(
        masks=dict(
            train_mask=train_mask_column_values,
            val_mask=val_mask_column_values,
            test_mask=test_mask_column_values
        ),
        node_ids=key_column_values,
        targets=target_column_values,
        features=features_values,
        features_columns=all_features_names,
    )

def read_edges_table_and_get_adgacency(mr_table, node_id_to_index_mapping: Mapping[str, int], client: yt.YtClient,
                                       _loading_metadata_path: str, loading_metadata: Dict[str, bool or int]) -> str:
    
    
    def append_edges_and_make_checkpoint(running_container: List[List[int]], row_number):
        with open("checkpoints/dataset/edges.csv", "a") as edges_file_handler:
            df_partial = pd.DataFrame(running_container).astype(np.int64)
            df_partial.to_csv(edges_file_handler, index=False, header=False)
        
        
        loading_metadata["edge_index_rows_loaded"] = row_number
        json.dump(loading_metadata, open(_loading_metadata_path, "w"))
        copy_out_to_snapshot("./", dump=True)
        
        print(f"Saved checkpoint at {row_number}-th row")
        
        del df_partial
    
    rows_already_loaded = loading_metadata["edge_index_rows_loaded"]
    yt_iterator = client.read_table(yt.TablePath(mr_table["table"], start_index=rows_already_loaded),                                                                       
                                format="json", 
                                unordered=False, 
                                enable_read_parallel=True,
                                raw=True,
                                )
    
    num_rows = get_row_count(mr_table["table"], client)
    
    running_container: List[List[int]] = []
    
    for i, row in enumerate(yt_iterator, rows_already_loaded+1):
        row = json.loads(row)
        try:
            start = node_id_to_index_mapping[row["source"]]
            end = node_id_to_index_mapping[row["target"]]
            
            running_container.append([start, end])
            
            if i % 10_000_000 == 0:
                print(f"Processed {i / 1_000_000}M/{num_rows / 1_000_000}M rows")
                gc.collect()
            
                if i % 100_000_000 == 0: # merge containers
                    append_edges_and_make_checkpoint(running_container, i)
                    running_container = []
                    gc.collect()


        except KeyError:
            print("Filtered edge with at least one end not presented in features dataframe")
        finally:
            del row

    if running_container:
        append_edges_and_make_checkpoint(running_container, i)
    
    print(f"Saved graph structure. Obtained edges: {rows_already_loaded+1}")
        
    return "checkpoints/dataset/edges.csv"

def make_client(yt_proxy: str = "hahn", max_thread_count: int = 4, enable: bool = True, token=None) -> yt.YtClient:
    from datetime import timedelta
    config = {
        "read_retries": {"enable": enable},
        "allow_receive_token_by_current_ssh_session": True,
        "table_writer": {"desired_chunk_size": 1024 * 1024 * 500},
        "concatenate_retries": {
            "enable": enable,
            "total_timeout": timedelta(minutes=128),
        },
        "write_retries": {"enable": enable, "count": 30},
    }
    if max_thread_count > 1:
        config["read_parallel"] = {
            "enable": True,
            "max_thread_count": max_thread_count,
        }
        config["write_parallel"] = {
            "enable": True,
            "max_thread_count": max_thread_count,
            "unordered": True,
        }
    return yt.YtClient(proxy=yt_proxy, config=config, token=token)

def get_row_count(path: yt.YPath, client: yt.YtClient) -> int:
    return yt.get_attribute(path=path, attribute="row_count", client=client)


def main_prepare_mr_tables(
    features_mr_table: Dict[str, str],
    edges_mr_table: Dict[str, str],
        
    features_table_loaded: bool,
    edge_index_rows_loaded: int,
    _loading_metadata_path: str,

    token=None,
    train_metadata: Optional[Dict[str, List[str]]] = None,
):
    print(f"{features_mr_table=}\n{edges_mr_table=}")
    
    client = make_client(features_mr_table["cluster"], max_thread_count=64, token=token)

    PARAMS_OUTPUT = {}
    
    # TODO check whether features are already processed and load them
    feature_columns_presented_in_train_df = None if train_metadata is None else train_metadata["features_columns"]
    
    
    loading_metadata = dict(features_table_loaded=features_table_loaded, edge_index_rows_loaded=edge_index_rows_loaded)
    
    if features_table_loaded:
        print("Loading already saved features array and other data information (stored in dictionary)")
        data_dict: Dict[str, np.ndarray or Dict[str, np.ndarray]] = joblib.load("./checkpoints/data_dict.pkl")
        
    else:
        data_dict: Dict[str, np.ndarray or Dict[str, np.ndarray]] = read_features_table(mr_table=features_mr_table, feature_columns_presented_in_train=feature_columns_presented_in_train_df, client=client)

        joblib.dump(data_dict, "./checkpoints/data_dict.pkl")
    
    print(f"Read train table, extracted features, train/val/test masks and all service columns from features dataframe ({SERVICE_COLUMNS_IN_FEATURES_TABLE})")
    
    _node_ids_to_index_mapping = dict(zip(data_dict["node_ids"], range(len(data_dict["node_ids"]))))
    print("Calculated node ids matching with their corresponding indices")

    loading_metadata["features_table_loaded"] = True
    
    # save checkpoint after loaded features and other data stuff:
    json.dump(loading_metadata, open(_loading_metadata_path, "w"))
    copy_out_to_snapshot("./", dump=True)
    
    print("Dumped data information to snapshot")
    
    
    # TODO check whether you should start from the beginning or not
    if loading_metadata.get("adjacency_loaded"):
        
        edges_file = "checkpoints/edges.csv"
    else:
        edges_file = read_edges_table_and_get_adgacency(mr_table=edges_mr_table, node_id_to_index_mapping=_node_ids_to_index_mapping, client=client,
                                                                        _loading_metadata_path=_loading_metadata_path, loading_metadata=loading_metadata)
        
        loading_metadata["adjacency_loaded"] = True
    
    json.dump(loading_metadata, open(_loading_metadata_path, "w"))
    copy_out_to_snapshot("./", dump=True)
    
    print("Dumped edges information to snapshot")

    
    PARAMS_OUTPUT["features"] = data_dict["features"]
    PARAMS_OUTPUT["targets"] = data_dict["targets"]

    PARAMS_OUTPUT["node_ids"] = data_dict["node_ids"]
    PARAMS_OUTPUT["node_indices"] = np.array([_node_ids_to_index_mapping[node_id] for node_id in PARAMS_OUTPUT["node_ids"]])
    PARAMS_OUTPUT["node_index_to_id_mapper"] = {node_index: node_id for node_id, node_index in _node_ids_to_index_mapping.items()}
    

    PARAMS_OUTPUT["masks"] = data_dict["masks"]
    PARAMS_OUTPUT["edges_file"] = edges_file


    PARAMS_OUTPUT["train_metadata"] = dict(features_columns=data_dict["features_columns"])
    


    return PARAMS_OUTPUT


if __name__ == "__main__":
    import os

    output = main_prepare_mr_tables(
        mr_tables=[
            {
                "cluster": "hahn",
                "table": "//home/yr/fvelikon/nirvana/c8e0052c-2996-4960-944f-b31fc5a8ca80/output1__gamlfOCgRqyq94sYvTIYtg",
            },
            {
                "cluster": "hahn",
                "table": "//home/yr/fvelikon/nirvana/5c3d379a-9604-4b97-a1d0-719238f7076f/output1__aoSo9RpQSQqumA-qBAjW5w",
            },
        ],
        token=os.environ.get("YT_TOKEN"),
    )
