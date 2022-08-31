import matplotlib.pyplot as plt
from PrintResults import getAllResult
import argparse

fig = plt.figure()
plt.ioff()


def plot_graph(dico, name, argdict):
    fig = plt.figure()
    plt.ioff()
    for key, value in dico.items():
        if key=="help":
            continue
        print(key, value)
        x=[]
        y=[]
        for split, acc in value.items():
            print(split, acc)
            x.append(split)
            y.append(acc)
        plt.plot(x, y, label=key)
    plt.xlabel('percent of the dataset added')
    plt.ylabel('accuracy on the validation set')
    plt.legend()
    plt.title(f"Accuracy vs split for various {name}")
    plt.savefig(f'Graphes/{argdict["algo"]}_{name}VsSplit_LastValue.png')
    plt.close(fig)
    #CONCLUSION: OVERFITTING

#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='VAE for data augmentation')
#     # General arguments on training
#     parser.add_argument('--dataset', type=str, default='SST-2',
#                         help="dataset you want to run the process on. Includes SST2, TREC6")
#     parser.add_argument('--classifier', type=str, default='lstm',
#                         help="classifier you want to use. Includes bert, lstm, jiantLstm or xgboost")
#     parser.add_argument('--computer', type=str, default='home',
#                         help="Whether you run at home or at iro. Automatically changes the base path")
#     parser.add_argument('--split', type=float, default=1.0, help='percent of the dataset added')
#     parser.add_argument('--retrain', action='store_true', help='whether to retrain the VAE or not')
#     parser.add_argument('--rerun', action='store_true',
#                         help='whether to rerun knowing that is has already been ran')
#     parser.add_argument('--dataset_size', type=int, default=10,
#                         help='number of example in the original dataset. If 0, use the entire dataset')
#     parser.add_argument('--algo', type=str, default='EDA',
#                         help='data augmentation algorithm to use, includes, EDA, W2V, GAN, VAE, and CVAE')
#     parser.add_argument('--random_seed', type=int, default=7, help='Random seed ')
#
#     # VAE specific arguments
#     parser.add_argument('--nb_epoch_algo', type=int, default=10, help="Number of epoch for which to run the VAE")
#
#     # Classifier specific arguments
#     parser.add_argument('--nb_epochs_lstm', type=int, default=5, help="Number of epoch to train the lstm with")
#     parser.add_argument('--dropout_classifier', type=int, default=0.3, help="dropout parameter of the classifier")
#     parser.add_argument('--hidden_size_classifier', type=int, default=128,
#                         help="dropout parameter of the classifier")
#     parser.add_argument('--latent_size', type=int, default=10,
#                         help="Size of the latent space for the VAE, CVAE, or GANs")
#
#     # Experiments
#     parser.add_argument('--test_dss_vs_split', action='store_true',
#                         help='test the influence of the dataset size and ')
#     parser.add_argument('--test_epochLstm_vs_split', action='store_true',
#                         help='test the influence of the number of epoch for the lstm classifier')
#     parser.add_argument('--test_dropoutLstm_vs_split', action='store_true',
#                         help='test the influence of the dropout for the lstm classifier')
#     parser.add_argument('--test_hsLstm_vs_split', action='store_true',
#                         help='test the influence of the dropout for the lstm classifier')
#     parser.add_argument('--test_latent_size_vs_split', action='store_true',
#                         help='test the influence of the latent size')
#     parser.add_argument('--test_randomSeed', action='store_true',
#                         help='test the influence of the random seed for LSTM')
#     parser.add_argument('--test_hidden_size_algo_vs_split', action='store_true', help='test the influence of the hidden size on the algorithm')
#     parser.add_argument('--run_ds_split', action='store_true',
#                         help='Run all DS and splits with specified arguments')
#
#     args = parser.parse_args()
#
#     argsdict = args.__dict__
#     if argsdict['computer'] == 'home':
#         argsdict['pathDataOG'] = "/media/frederic/DAGlue/data"
#         argsdict['pathDataAdd'] = "/media/frederic/DAGlue"
#     elif argsdict['computer'] == 'labo':
#         argsdict['pathDataOG'] = "/data/rali5/Tmp/piedboef/data/jiantData/OG/OG"
#         argsdict['pathDataAdd'] = "/u/piedboef/Documents/DAGlue"
#     # Create directories for the runs
#     # Categories
#     if argsdict['dataset'] == "SST-2":
#         categories = ["neg", "pos"]
#     elif argsdict['dataset'] == "TREC6":
#         categories = ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]
#     argsdict['categories'] = categories
#
#     # json.dump(argsdict, open(f"/media/frederic/DAGlue/SentenceVAE/GeneratedData/{argsdict['dataset']}/{argsdict['dataset_size']}/parameters.json", "w"))
#
#     print("=================================================================================================")
#     print(argsdict)
#     print("=================================================================================================")
#
#     if argsdict['test_latent_size_vs_split']:
#         dico = {}
#         argsdict['dataset_size'] = 500
#         for latent_size in [10, 20, 50, 100]:
#             dicoTemp = {}
#             argsdict['retrain'] = True
#             for split in [0, 0.2, 0.4, 0.6, 0.8, 1]:
#                 print(f"Latent Size {latent_size}, split {split}")
#                 argsdict['latent_size'] = latent_size
#                 argsdict['split'] = split
#                 # createFoldersEDA(argsdict)
#                 acc_dev = getAllResult(argsdict)
#                 dicoTemp[split] = acc_dev
#             dico[latent_size] = dicoTemp
#         plot_graph(dico, "LatentSize", argsdict)
#     elif argsdict['test_hidden_size_algo_vs_split']:
#         dico = {}
#         argsdict['dataset_size'] = 5000
#         argsdict['latent_size'] = 10
#         # argsdict['split'] = 0.5
#         for hidden_size in [10, 20, 50, 100]:
#             dicoTemp={}
#             argsdict['retrain'] = True
#             for split in [0, 0.2, 0.4, 0.6, 0.8, 1]:
#                 print(f"Hidden Size {hidden_size}, split {split}")
#                 argsdict['hidden_size_algo']=hidden_size
#                 argsdict['split']=split
#                 # createFoldersEDA(argsdict)
#                 acc_val=getAllResult(argsdict)
#                 dicoTemp[split]=acc_val
#             dico[hidden_size]=dicoTemp
#         plot_graph(dico, "HiddenSizeAlgo")
#     elif argsdict['test_randomSeed']:
#         dico = {'help': 'key is random seed, then key inside is split and value is a tuple of train val accuracy (lists)'}
#         argsdict['dataset_size'] = 5000
#         argsdict['nb_epoch_lstm']=5
#         argsdict['dropout_classifier']=0.3
#         for random_seed in [7,3,9,13,100,500]:
#             dicoTemp={}
#             for split in [0, 0.2, 0.4, 0.6, 0.8, 1]:
#                 print(f"Random Seed {random_seed}, split {split}")
#                 argsdict['random_seed']=random_seed
#                 argsdict['split']=split
#                 # argsdict['retrain']=False
#                 # createFoldersEDA(argsdict)
#                 acc_dev = getAllResult(argsdict)
#                 dicoTemp[split] = acc_dev
#             dico[random_seed]=dicoTemp
#             plot_graph(dico, "RandomSeed")
# # Random seeds: 3,7,9,13,100, -500, 512, 12, 312, 888