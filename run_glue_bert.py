import os
# import jiant.proj.simple
import jiant.proj.simple.runscript as simple_run
import jiant.scripts.download_data.runscript as downloader
from process_data import get_results_bert
import argparse
# See https://github.com/nyu-mll/jiant/blob/master/guides/tasks/supported_tasks.md for supported tasks
# TASK_NAME = ["mrpc", "sst", "stsb", "wnli", "cola", "glue_diagnostics", "mnli", "mnli_mismatched", "qnli", "qqp", "rte", ]

def run_bert(argdict):
    print(argdict)
    if argdict['dataset']=="SST-2":
        TASK_NAME="sst"
    elif argdict['dataset']=="TREC6":
        TASK_NAME="trec"
    elif argdict['dataset']=="FakeNews":
        TASK_NAME="FakeNews"
    elif argdict['dataset']=="QNLI":
        TASK_NAME="qnli"
    elif argdict['dataset']=="Irony":
        TASK_NAME="Irony"
    elif argdict['dataset']=="IronyB":
        TASK_NAME="IronyB"
    elif argdict['dataset']=="Subj":
        TASK_NAME="Subj"
    else:
        raise ValueError("Task Not Found")

    # See https://huggingface.co/models for supported models
    MODEL_TYPE = "bert-base-uncased"

    RUN_NAME = f"simple_{TASK_NAME}_{MODEL_TYPE}"
    EXP_DIR = f"{argdict['pathDataAdd']}/content/exp"
    #Data dir needs to be a global path, probably due to the
    DATA_DIR = f"{argdict['pathDataAdd']}/data/bert/"
    # EXP_DIR='/data/rali5/Tmp/piedboef/data/GLUE'
    # DATA_DIR='/data/rali5/Tmp/piedboef/data/GLUE/data'


    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(EXP_DIR, exist_ok=True)

    # downloader.download_data(['qnli'], DATA_DIR)



    if argdict['nb_epochs_lstm'] in [20, 8, 15, 10, 11]:
        num_ep=argdict['nb_epochs_lstm']
    else:
        num_ep=4

    args = simple_run.RunConfiguration(
        run_name=RUN_NAME,
        exp_dir=EXP_DIR,
        data_dir=DATA_DIR,
        model_type=MODEL_TYPE,
        train_tasks=TASK_NAME,
        val_tasks=TASK_NAME,
        test_tasks=TASK_NAME,
        train_batch_size=16,
        num_train_epochs=num_ep,
        seed=argdict['random_seed']
    )
    simple_run.run_simple(args)
    return get_results_bert(argdict)
