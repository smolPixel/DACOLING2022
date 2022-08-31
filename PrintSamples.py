import argparse, json
from process_data import *

def print_samples(argdict):
    found = False
    i=0
    while not found:
        path = f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{i}"
        # Open the folder. If it works then check if its the same experiment, else go to next one
        param = json.load(open(f"{path}/param.json", 'r'))
        if compareFiles(argdict, param, "gen"):
            print(f"Dataset found in Folder {i}")
            found = True
        else:
            i += 1
    #We found i, getting samples
    for cat in argdict['categories']:
        path = f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{i}/{cat}.txt"
        file=open(path, 'r').readlines()[:5]
        print(cat)
        print(file)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='VAE for data augmentation')
#     #General arguments on training
#     parser.add_argument('--dataset', type=str, default='SST-2', help="dataset you want to run the process on. Includes SST2, TREC6, FakeNews")
#     parser.add_argument('--classifier', type=str, default='lstm', help="classifier you want to use. Includes bert, lstm, jiantLstm or xgboost")
#     parser.add_argument('--computer', type=str, default='home', help="Whether you run at home or at iro. Automatically changes the base path")
#     parser.add_argument('--split', type=float, default=1.0, help='percent of the dataset added')
#     parser.add_argument('--retrain', action='store_true', help='whether to retrain the VAE or not')
#     parser.add_argument('--rerun', action='store_true', help='whether to rerun knowing that is has already been ran')
#     parser.add_argument('--dataset_size', type=int, default=10, help='number of example in the original dataset. If 0, use the entire dataset')
#     parser.add_argument('--algo', type=str, default='EDA', help='data augmentation algorithm to use, includes, EDA, W2V, GAN, VAE, CVAE, CBERT, and GPT')
#     parser.add_argument('--random_seed', type=int, default=7, help='Random seed ')
#
#     #Algo specific arguments
#     parser.add_argument('--nb_epoch_algo', type=int, default=10, help="Number of epoch for which to run the algo when applicable")
#     parser.add_argument('--hidden_size_algo', type=int, default=256, help="Hidden Size for the algo when applicable (VAE, CVAE, GAN, CatGAN)")
#
#     #Classifier specific arguments
#     parser.add_argument('--nb_epochs_lstm', type=int, default=5, help="Number of epoch to train the lstm with")
#     parser.add_argument('--dropout_classifier', type=int, default=0.3, help="dropout parameter of the classifier")
#     parser.add_argument('--hidden_size_classifier', type=int, default=128, help="dropout parameter of the classifier")
#     parser.add_argument('--latent_size', type=int, default=10, help="Size of the latent space for the VAE, CVAE, or GANs")
#
#     #Experiments
#     parser.add_argument('--test_dss_vs_split', action='store_true', help='test the influence of the dataset size and ')
#     parser.add_argument('--test_epochLstm_vs_split', action='store_true', help='test the influence of the number of epoch for the lstm classifier')
#     parser.add_argument('--test_dropoutLstm_vs_split', action='store_true', help='test the influence of the dropout for the lstm classifier')
#     parser.add_argument('--test_hsLstm_vs_split', action='store_true', help='test the influence of the dropout for the lstm classifier')
#     parser.add_argument('--test_latent_size_vs_split', action='store_true', help='test the influence of the latent size')
#     parser.add_argument('--test_randomSeed', action='store_true', help='test the influence of the random seed for LSTM')
#     parser.add_argument('--run_ds_split', action='store_true', help='Run all DS and splits with specified arguments')
#     parser.add_argument('--test_one_shot', action='store_true', help='test a one shot combination')
#     parser.add_argument('--test_hidden_size_algo_vs_split', action='store_true', help='test the influence of the hidden size on the algorithm')
#
#     args = parser.parse_args()
#
#     argsdict = args.__dict__
#     if argsdict['computer']=='home':
#         argsdict['pathDataOG'] = "/media/frederic/DAGlue/data"
#         argsdict['pathDataAdd'] = "/media/frederic/DAGlue"
#     elif argsdict['computer']=='labo':
#         argsdict['pathDataOG'] = "/data/rali5/Tmp/piedboef/data/jiantData/OG/OG"
#         argsdict['pathDataAdd'] = "/u/piedboef/Documents/DAGlue"
#     # Create directories for the runs
#     # Categories
#     if argsdict['dataset'] == "SST-2":
#         categories = ["neg", "pos"]
#     elif argsdict['dataset'] == "TREC6":
#         categories = ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]
#     elif argsdict['dataset'] == "FakeNews":
#         categories = ["Real", "Fake"]
#     argsdict['categories'] = categories
#     if argsdict['test_latent_size_vs_split']:
#         dico = {
#             'help': 'key is latent size, then key inside is split and value is a tuple of train val accuracy (lists)'}
#         argsdict['dataset_size'] = 500
#         # argsdict['split'] = 0.5
#         for latent_size in [10, 20, 50, 100]:
#             dicoTemp = {}
#             print(f"Latent Size {latent_size}")
#             argsdict['latent_size'] = latent_size
#             # createFoldersEDA(argsdict)
#             main(argsdict)