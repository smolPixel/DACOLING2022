"""Functions for modifying things to ensure that everything still runs smoothly"""
import os, json

def getOnlyDirectory(root):
    """Return a list of all direct subdirectories in root"""
    print(root)
    return [ds for ds in os.listdir(root) if os.path.isdir(f"{root}/{ds}")]

def add_param(param, algos):
    """Add the batch size to all experiments and generated config files"""
    # algos={"EDA":25, "W2V":25, "VAE":32, "GAN":5, "CVAE":32, "CATGAN":5, "CBERT":32, "GPT":32}
    # bs=[25, 25, 32, 5, 32, 5, 32, 32]
    Root="Experiments/Record"
    for datasets in os.listdir(Root):
        if datasets!="TREC6":
            continue
        for algo in os.listdir(f"{Root}/{datasets}"):
            for classifier in os.listdir(f"{Root}/{datasets}/{algo}"):
                for experiment in os.listdir(f"{Root}/{datasets}/{algo}/{classifier}"):
                    path=f"{Root}/{datasets}/{algo}/{classifier}/{experiment}/param.json"
                    try:
                        file=json.load(open(path))
                        print(file)
                        print(algos[algo])
                        if param in file:
                            print("param already in file")
                            continue
                        print("adding parameter")
                        file[param]=algos[algo]
                        with open(path, "w") as f:
                            json.dump(file, f)
                    except:
                        pass
    Root = "GeneratedData"
    for algo in os.listdir(Root):
        datasets= getOnlyDirectory(f"{Root}/{algo}")
        for dataset in datasets:
            listedss = getOnlyDirectory(f"{Root}/{algo}/{dataset}")
            for dataset_size in listedss:
                for experiment in os.listdir(f"{Root}/{algo}/{dataset}/{dataset_size}"):
                    path=f"{Root}/{algo}/{dataset}/{dataset_size}/{experiment}/param.json"
                    try:
                        file = json.load(open(path))
                        # print(file)
                        # print(algos[algo])
                        if param in file:
                            continue
                        file[param] = algos[algo]
                        with open(path, "w") as f:
                            json.dump(file, f)
                    except:
                        pass

# add_param('bs_algo', {"EDA":25, "W2V":25, "VAE":32, "GAN":5, "CVAE":32, "CATGAN":5, "CBERT":32, "GPT":32})
# add_param('max_length', {"EDA":0, "W2V":0, "VAE":0, "GAN":0, "CVAE":0, "CATGAN":0, "CBERT":0, "GPT":0})
add_param('starting_set_seed', {"EDA":0, "W2V":0, "VAE":0, "GAN":0, "CVAE":0, "CATGAN":0, "CBERT":0, "GPT":0})
# add_param('vary_starting_set', {"EDA":False, "W2V":False, "VAE":False, "GAN":False, "CVAE":False, "CATGAN":False, "CBERT":False, "GPT":False})