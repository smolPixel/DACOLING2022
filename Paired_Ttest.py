import subprocess, json
import numpy as np
from scipy.stats import f
#Paired ttest for everything For each algo, classifier and split, run a paired T-test between the baseline and the augmented results
from scipy.stats import ttest_rel
def run_external_process(process):
    output, error = process.communicate()
    if process.returncode != 0:
        raise SystemError
    return output, error

def mult_paired_t_test(base, aug):
    #shape is [num examples, num features]. So here it should be [30, 5]
    diff=base-aug
    mean=np.mean(diff, axis=0)
    stdev=np.std(diff, axis=0, ddof=1)
    cov=np.cov(np.transpose(diff))
    n=diff.shape[0]
    k=diff.shape[1]
    Goal=np.zeros(k)
    T2=n*np.matmul(np.matmul(mean-Goal, np.linalg.inv(cov)), mean-Goal)
    F=T2*(n-k)/(k*(n-1))
    df1=k
    df2=n-k
    p=1-f.cdf(F, df1, df2)
    return p

#TODO ADD FAKENEWS W2V 0
#TODO W2V
commands={
    'EDA':{100:['python3 run_algo.py --dataset_size 100 --algo EDA --split 1 --bs_algo 25',
                'python3 run_algo.py --dataset_size 100 --algo EDA --split 1 --bs_algo 25 --dataset TREC6',
                'python3 run_algo.py --dataset_size 100 --algo EDA --split 1 --bs_algo 25 --dataset FakeNews',
                'python3 run_algo.py --dataset_size 100 --algo EDA --split 1 --bs_algo 25 --dataset Irony',
                'python3 run_algo.py --dataset_size 100 --algo EDA --split 1 --bs_algo 25 --dataset IronyB'],
                 500:['python3 run_algo.py --dataset_size 500 --algo EDA --split 1 --bs_algo 25',
                      'python3 run_algo.py --dataset_size 500 --algo EDA --split 1 --bs_algo 25 --dataset TREC6', 'python3 run_algo.py --dataset_size 500 --algo EDA --split 1 --bs_algo 25 --dataset FakeNews', 'python3 run_algo.py --dataset_size 500 --algo EDA --split 1 --bs_algo 25 --dataset Irony', 'python3 run_algo.py --dataset_size 500 --algo EDA --split 1 --bs_algo 25 --dataset IronyB'],
                 1000:['python3 run_algo.py --dataset_size 1000 --algo EDA --split 1 --bs_algo 25',
                       'python3 run_algo.py --dataset_size 1000 --algo EDA --split 1 --bs_algo 25 --dataset TREC6', 'python3 run_algo.py --dataset_size 1000 --algo EDA --split 1 --bs_algo 25 --dataset FakeNews', 'python3 run_algo.py --dataset_size 1000 --algo EDA --split 1 --bs_algo 25 --dataset Irony', 'python3 run_algo.py --dataset_size 1000 --algo EDA --split 1 --bs_algo 25 --dataset IronyB'],
                 0:['python3 run_algo.py --dataset_size 0 --algo EDA --split 1 --bs_algo 25', 'python3 run_algo.py --dataset_size 0 --algo EDA --split 1 --bs_algo 25 --dataset TREC6', 'python3 run_algo.py --dataset_size 0 --algo EDA --split 1 --bs_algo 25 --dataset FakeNews', 'python3 run_algo.py --dataset_size 0 --algo EDA --split 1 --bs_algo 25 --dataset Irony', 'python3 run_algo.py --dataset_size 0 --algo EDA --split 1 --bs_algo 25 --dataset IronyB']},
          'W2V': {100: ['python3 run_algo.py --dataset_size 100 --algo W2V --split 1 --bs_algo 25',
                        'python3 run_algo.py --dataset_size 100 --algo W2V --split 1 --bs_algo 25 --dataset TREC6',
                        'python3 run_algo.py --dataset_size 100 --algo W2V --split 1 --bs_algo 25 --dataset FakeNews',
                        'python3 run_algo.py --dataset_size 100 --algo W2V --split 1 --bs_algo 25 --dataset Irony',
                        'python3 run_algo.py --dataset_size 100 --algo W2V --split 1 --bs_algo 25 --dataset IronyB'],
                  500: ['python3 run_algo.py --dataset_size 500 --algo W2V --split 1 --bs_algo 25',
                        'python3 run_algo.py --dataset_size 500 --algo W2V --split 1 --bs_algo 25 --dataset TREC6',
                        'python3 run_algo.py --dataset_size 500 --algo W2V --split 1 --bs_algo 25 --dataset FakeNews',
                        'python3 run_algo.py --dataset_size 500 --algo W2V --split 1 --bs_algo 25 --dataset Irony',
                        'python3 run_algo.py --dataset_size 500 --algo W2V --split 1 --bs_algo 25 --dataset IronyB'],
                  1000: ['python3 run_algo.py --dataset_size 1000 --algo W2V --split 1 --bs_algo 25',
                         'python3 run_algo.py --dataset_size 1000 --algo W2V --split 1 --bs_algo 25 --dataset TREC6',
                         'python3 run_algo.py --dataset_size 1000 --algo W2V --split 1 --bs_algo 25 --dataset FakeNews',
                         'python3 run_algo.py --dataset_size 1000 --algo W2V --split 1 --bs_algo 25 --dataset Irony',
                         'python3 run_algo.py --dataset_size 1000 --algo W2V --split 1 --bs_algo 25 --dataset IronyB'],
                  0: ['python3 run_algo.py --dataset_size 0 --algo W2V --split 1 --bs_algo 25',
                      'python3 run_algo.py --dataset_size 0 --algo W2V --split 1 --bs_algo 25 --dataset TREC6',
                      'python3 run_algo.py --dataset_size 0 --algo W2V --split 1 --bs_algo 25 --dataset FakeNews',
                      'python3 run_algo.py --dataset_size 0 --algo W2V --split 1 --bs_algo 25 --dataset Irony',
                      'python3 run_algo.py --dataset_size 0 --algo W2V --split 1 --bs_algo 25 --dataset IronyB']
                  },
        'VAE': {100:['python3 run_algo.py --algo VAE --split 1 --dataset_size 100 --x0 500 --latent_size 5 --hidden_size_algo 1024 --nb_epoch_algo 20 --dropout_algo 0.5 --word_dropout 0.6 --unidir_algo',
                       'python3 run_algo.py --algo VAE --split 1 --dataset_size 100 --x0 600 --latent_size 5 --hidden_size_algo 1024 --nb_epoch_algo 20 --dropout_algo 0.5 --word_dropout 0.6 --unidir_algo --dataset TREC6 ',
                       'python3 run_algo.py --algo VAE --split 1 --dataset_size 100 --x0 600 --latent_size 5 --hidden_size_algo 1024 --nb_epoch_algo 20 --dropout_algo 0.5 --word_dropout 0.6 --unidir_algo --bs_algo 32 --dataset FakeNews',
                       'python3 run_algo.py --algo VAE --split 1 --dataset_size 100 --x0 625 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --unidir_algo --bs_algo 32 --dataset Irony',
                       'python3 run_algo.py --algo VAE --split 1 --dataset_size 100 --x0 800 --latent_size 5 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --unidir_algo --dataset IronyB'],
                  500: [
                      'python3 run_algo.py --algo VAE --split 1 --dataset_size 500 --x0 15 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --k 0.00025',
                      'python3 run_algo.py --algo VAE --split 1 --dataset_size 500 --x0 15 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --k 0.00025 --dataset TREC6 ',
                      'python3 run_algo.py --algo VAE --split 1 --dataset_size 500 --x0 15 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --k 0.00025  --dataset FakeNews',
                      'python3 run_algo.py --algo VAE --split 1 --dataset_size 500 --x0 15 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --k 0.00025 --dataset Irony',
                      'python3 run_algo.py --algo VAE --split 1 --dataset_size 500 --x0 15 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --k 0.00025 --dataset IronyB'],
                  1000: [
                      'python3 run_algo.py --algo VAE --split 1 --dataset_size 1000 --x0 15 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --k 0.00025',
                      'python3 run_algo.py --algo VAE --split 1 --dataset_size 1000 --x0 15 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --k 0.00025 --dataset TREC6 ',
                      'python3 run_algo.py --algo VAE --split 1 --dataset_size 1000 --x0 15 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --k 0.00025 --dataset FakeNews',
                      'python3 run_algo.py --algo VAE --split 1 --dataset_size 1000 --x0 15 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --k 0.00025 --dataset Irony',
                      'python3 run_algo.py --algo VAE --split 1 --dataset_size 1000 --x0 15 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --k 0.00025 --dataset IronyB'],
                  0: [
                      'python3 run_algo.py --algo VAE --split 1 --dataset_size 0 --x0 650 --latent_size 5 --hidden_size_algo 1024 --nb_epoch_algo 20 --dropout_algo 0.5 --word_dropout 0.6 --unidir_algo',
                      'python3 run_algo.py --algo VAE --split 1 --dataset_size 0 --x0 600 --latent_size 5 --hidden_size_algo 1024 --nb_epoch_algo 20 --dropout_algo 0.5 --word_dropout 0.6 --unidir_algo --dataset TREC6 ',
                      'python3 run_algo.py --algo VAE --split 1 --dataset_size 0 --x0 600 --latent_size 5 --hidden_size_algo 1024 --nb_epoch_algo 20 --dropout_algo 0.5 --word_dropout 0.6 --unidir_algo --bs_algo 32 --dataset FakeNews',
                      'python3 run_algo.py --algo VAE --split 1 --dataset_size 0 --x0 625 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --unidir_algo --bs_algo 32 --dataset Irony',
                      'python3 run_algo.py --algo VAE --split 1 --dataset_size 0 --x0 800 --latent_size 5 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --unidir_algo --dataset IronyB']
                  },
            'VAE_EncDec': {100:['python3 run_algo.py --algo VAE_EncDec --split 1 --dataset_size 1000 --x0 15 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --k 0.00025',
                                'python3 run_algo.py --algo VAE_EncDec --split 1 --dataset_size 100 --x0 100 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --dataset TREC6',
                                'python3 run_algo.py --algo VAE_EncDec --split 1 --dataset_size 100 --x0 600 --latent_size 5 --hidden_size_algo 1024 --nb_epoch_algo 20 --dropout_algo 0.5 --word_dropout 0.6 --unidir_algo --bs_algo 32 --dataset FakeNews ',
                                'python3 run_algo.py --algo VAE_EncDec --split 1 --dataset_size 100 --x0 625 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --unidir_algo --bs_algo 32 --dataset Irony',
                                'python3 run_algo.py --algo VAE_EncDec --split 1 --dataset_size 100 --x0 800 --latent_size 5 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --unidir_algo --dataset IronyB'],
                           500:['python3 run_algo.py --algo VAE_EncDec --split 1 --dataset_size 500 --x0 15 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --k 0.00025',
                                'python3 run_algo.py --algo VAE_EncDec --split 1 --dataset_size 500 --x0 15 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --k 0.00025 --dataset TREC6',
                                'python3 run_algo.py --algo VAE_EncDec --split 1 --dataset_size 500 --x0 15 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --k 0.00025 --bs_algo 32 --dataset FakeNews ',
                                'python3 run_algo.py --algo VAE_EncDec --split 1 --dataset_size 500 --x0 15 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --k 0.00025 --bs_algo 32 --dataset Irony',
                                'python3 run_algo.py --algo VAE_EncDec --split 1 --dataset_size 500 --x0 15 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --k 0.00025 --dataset IronyB'],
                          1000:['python3 run_algo.py --algo VAE_EncDec --split 1 --dataset_size 1000 --x0 15 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --k 0.00025',
                                'python3 run_algo.py --algo VAE_EncDec --split 1 --dataset_size 1000 --x0 15 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --k 0.00025 --dataset TREC6',
                                'python3 run_algo.py --algo VAE_EncDec --split 1 --dataset_size 1000 --x0 15 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --k 0.00025 --dataset FakeNews ',
                                'python3 run_algo.py --algo VAE_EncDec --split 1 --dataset_size 1000 --x0 15 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --k 0.00025  --dataset Irony',
                                'python3 run_algo.py --algo VAE_EncDec --split 1 --dataset_size 1000 --x0 15 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --k 0.00025  --dataset IronyB'],
                          0:['python3 run_algo.py --algo VAE_EncDec --split 1 --dataset_size 0 --x0 100 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6',
                                'python3 run_algo.py --algo VAE_EncDec --split 1 --dataset_size 0 --x0 100 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 20 --dropout_algo 0.5 --word_dropout 0.6 --dataset TREC6',
                                'python3 run_algo.py --algo VAE_EncDec --split 1 --dataset_size 0 --x0 600 --latent_size 5 --hidden_size_algo 1024 --nb_epoch_algo 20 --dropout_algo 0.5 --word_dropout 0.6 --unidir_algo --bs_algo 32 --dataset FakeNews ',
                                'python3 run_algo.py --algo VAE_EncDec --split 1 --dataset_size 0 --x0 625 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --unidir_algo --bs_algo 32 --dataset Irony',
                                'python3 run_algo.py --algo VAE_EncDec --split 1 --dataset_size 0 --x0 800 --latent_size 5 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --unidir_algo --dataset IronyB'],
                    },
          'GAN': {100: ['python3 run_algo.py --dataset_size 100 --algo GAN --split 1 --nb_epoch_algo 5 --bs_algo 5',
                        'python3 run_algo.py --dataset_size 100 --algo GAN --split 1 --nb_epoch_algo 5 --bs_algo 5 --dataset TREC6 ',
                        'python3 run_algo.py --dataset_size 100 --algo GAN --split 1 --nb_epoch_algo 5 --bs_algo 25 --dataset FakeNews',
                        'python3 run_algo.py --dataset_size 100 --algo GAN --split 1 --nb_epoch_algo 10 --bs_algo 25  --dataset Irony',
                        'python3 run_algo.py --dataset_size 100 --algo GAN --split 1 --nb_epoch_algo 5 --bs_algo 5 --dataset IronyB'],
                  500: ['python3 run_algo.py --dataset_size 500 --algo GAN --split 1 --nb_epoch_algo 5 --bs_algo 5',
                        'python3 run_algo.py --dataset_size 500 --algo GAN --split 1 --nb_epoch_algo 5 --bs_algo 25 --dataset TREC6 ',
                        'python3 run_algo.py --dataset_size 500 --algo GAN --split 1 --nb_epoch_algo 5 --bs_algo 25 --dataset FakeNews',
                        'python3 run_algo.py --dataset_size 500 --algo GAN --split 1 --nb_epoch_algo 10 --bs_algo 64  --dataset Irony',
                        'python3 run_algo.py --dataset_size 500 --algo GAN --split 1 --nb_epoch_algo 5 --bs_algo 25 --dataset IronyB'],
                  1000: ['python3 run_algo.py --dataset_size 1000 --algo GAN --split 1 --nb_epoch_algo 5 --bs_algo 64',
                        'python3 run_algo.py --dataset_size 1000 --algo GAN --split 1 --nb_epoch_algo 5 --bs_algo 64 --dataset TREC6 ',
                        'python3 run_algo.py --dataset_size 1000 --algo GAN --split 1 --nb_epoch_algo 5 --bs_algo 64 --dataset FakeNews',
                        'python3 run_algo.py --dataset_size 1000 --algo GAN --split 1 --nb_epoch_algo 10 --bs_algo 64  --dataset Irony',
                        'python3 run_algo.py --dataset_size 1000 --algo GAN --split 1 --nb_epoch_algo 5 --bs_algo 64 --dataset IronyB'],
                  0: ['python3 run_algo.py --dataset_size 0 --algo GAN --split 1 --nb_epoch_algo 5 --bs_algo 128',
                        'python3 run_algo.py --dataset_size 0 --algo GAN --split 1 --nb_epoch_algo 5 --bs_algo 64 --dataset TREC6 ',
                        'python3 run_algo.py --dataset_size 0 --algo GAN --split 1 --nb_epoch_algo 5 --bs_algo 64 --dataset FakeNews',
                        'python3 run_algo.py --dataset_size 0 --algo GAN --split 1 --nb_epoch_algo 10 --bs_algo 64  --dataset Irony',
                        'python3 run_algo.py --dataset_size 0 --algo GAN --split 1 --nb_epoch_algo 5 --bs_algo 64 --dataset IronyB']
                  },
          'CVAE': {100:['python3 run_algo.py --algo CVAE --split 1 --dataset_size 100 --unidir_algo --x0 700 --latent_size 5 --hidden_size_algo 1024 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6',
                        'python3 run_algo.py --algo CVAE --split 1 --dataset_size 100 --unidir_algo --x0 500 --latent_size 5 --hidden_size_algo 1024 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --bs_algo 32 --dataset TREC6',
                        'python3 run_algo.py --algo CVAE --split 1 --dataset_size 100 --unidir_algo --x0 500 --latent_size 5 --hidden_size_algo 1024 --nb_epoch_algo 20 --dropout_algo 0.5 --word_dropout 0.6 --dataset FakeNews',
                        'python3 run_algo.py --algo CVAE --split 1 --dataset_size 100 --unidir_algo --x0 625 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --dataset Irony',
                        'python3 run_algo.py --algo CVAE --split 1 --dataset_size 100 --unidir_algo --x0 800 --latent_size 5 --hidden_size_algo 1024 --nb_epoch_algo 20 --dropout_algo 0.5 --word_dropout 0.6 --dataset IronyB'],
                   500:['python3 run_algo.py --algo CVAE --split 1 --dataset_size 500 --unidir_algo --x0 700 --latent_size 5 --hidden_size_algo 1024 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6',
                        'python3 run_algo.py --algo CVAE --split 1 --dataset_size 500 --unidir_algo --x0 500 --latent_size 5 --hidden_size_algo 1024 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --bs_algo 32 --dataset TREC6',
                        'python3 run_algo.py --algo CVAE --split 1 --dataset_size 500 --unidir_algo --x0 500 --latent_size 5 --hidden_size_algo 1024 --nb_epoch_algo 20 --dropout_algo 0.5 --word_dropout 0.6 --dataset FakeNews',
                        'python3 run_algo.py --algo CVAE --split 1 --dataset_size 500 --unidir_algo --x0 625 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --dataset Irony',
                        'python3 run_algo.py --algo CVAE --split 1 --dataset_size 500 --unidir_algo --x0 800 --latent_size 5 --hidden_size_algo 1024 --nb_epoch_algo 20 --dropout_algo 0.5 --word_dropout 0.6 --dataset IronyB'],
                   1000:['python3 run_algo.py --algo CVAE --split 1 --dataset_size 1000 --unidir_algo --x0 700 --latent_size 5 --hidden_size_algo 1024 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6',
                        'python3 run_algo.py --algo CVAE --split 1 --dataset_size 1000 --unidir_algo --x0 500 --latent_size 5 --hidden_size_algo 1024 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --bs_algo 32 --dataset TREC6',
                        'python3 run_algo.py --algo CVAE --split 1 --dataset_size 1000 --unidir_algo --x0 500 --latent_size 5 --hidden_size_algo 1024 --nb_epoch_algo 20 --dropout_algo 0.5 --word_dropout 0.6 --dataset FakeNews',
                        'python3 run_algo.py --algo CVAE --split 1 --dataset_size 1000 --unidir_algo --x0 625 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --dataset Irony',
                        'python3 run_algo.py --algo CVAE --split 1 --dataset_size 1000 --unidir_algo --x0 800 --latent_size 5 --hidden_size_algo 1024 --nb_epoch_algo 20 --dropout_algo 0.5 --word_dropout 0.6 --dataset IronyB'],
                   0:['python3 run_algo.py --algo CVAE --split 1 --dataset_size 0 --unidir_algo --x0 700 --latent_size 5 --hidden_size_algo 1024 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6',
                        'python3 run_algo.py --algo CVAE --split 1 --dataset_size 0 --unidir_algo --x0 500 --latent_size 5 --hidden_size_algo 1024 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --bs_algo 32 --dataset TREC6',
                        'python3 run_algo.py --algo CVAE --split 1 --dataset_size 0 --unidir_algo --x0 500 --latent_size 5 --hidden_size_algo 1024 --nb_epoch_algo 20 --dropout_algo 0.5 --word_dropout 0.6 --dataset FakeNews',
                        'python3 run_algo.py --algo CVAE --split 1 --dataset_size 0 --unidir_algo --x0 625 --latent_size 10 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --dataset Irony',
                        'python3 run_algo.py --algo CVAE --split 1 --dataset_size 0 --unidir_algo --x0 800 --latent_size 5 --hidden_size_algo 1024 --nb_epoch_algo 20 --dropout_algo 0.5 --word_dropout 0.6 --dataset IronyB']
                   },
          'CATGAN': {100:['python3 run_algo.py --dataset_size 100 --algo CATGAN --split 1 --nb_epoch_algo 5 --bs_algo 5',
                          'python3 run_algo.py --dataset_size 100 --algo CATGAN --split 1 --nb_epoch_algo 15 --bs_algo 5 --dataset TREC6',
                          'python3 run_algo.py --dataset_size 100 --algo CATGAN --split 1 --nb_epoch_algo 15 --bs_algo 5 --dataset FakeNews',
                          'python3 run_algo.py --dataset_size 100 --algo CATGAN --split 1 --nb_epoch_algo 15 --bs_algo 5 --dataset Irony ',
                          'python3 run_algo.py --dataset_size 100 --algo CATGAN --split 1 --nb_epoch_algo 15 --bs_algo 5 --dataset IronyB '],
                     500:['python3 run_algo.py --dataset_size 500 --algo CATGAN --split 1 --nb_epoch_algo 5 --bs_algo 5',
                          'python3 run_algo.py --dataset_size 500 --algo CATGAN --split 1 --nb_epoch_algo 15 --bs_algo 5 --dataset TREC6',
                          'python3 run_algo.py --dataset_size 500 --algo CATGAN --split 1 --nb_epoch_algo 15 --bs_algo 5 --dataset FakeNews',
                          'python3 run_algo.py --dataset_size 500 --algo CATGAN --split 1 --nb_epoch_algo 15 --bs_algo 5 --dataset Irony ',
                          'python3 run_algo.py --dataset_size 500 --algo CATGAN --split 1 --nb_epoch_algo 15 --bs_algo 25 --dataset IronyB '],
                     1000:['python3 run_algo.py --dataset_size 1000 --algo CATGAN --split 1 --nb_epoch_algo 5 --bs_algo 5',
                          'python3 run_algo.py --dataset_size 1000 --algo CATGAN --split 1 --nb_epoch_algo 15 --bs_algo 5 --dataset TREC6',
                          'python3 run_algo.py --dataset_size 1000 --algo CATGAN --split 1 --nb_epoch_algo 15 --bs_algo 5 --dataset FakeNews',
                          'python3 run_algo.py --dataset_size 1000 --algo CATGAN --split 1 --nb_epoch_algo 15 --bs_algo 5 --dataset Irony ',
                          'python3 run_algo.py --dataset_size 1000 --algo CATGAN --split 1 --nb_epoch_algo 5 --bs_algo 25 --dataset IronyB '],
                     0:['python3 run_algo.py --dataset_size 0 --algo CATGAN --split 1 --nb_epoch_algo 5 --bs_algo 10',
                          'python3 run_algo.py --dataset_size 0 --algo CATGAN --split 1 --nb_epoch_algo 5 --bs_algo 10 --dataset TREC6',
                          'python3 run_algo.py --dataset_size 0 --algo CATGAN --split 1 --nb_epoch_algo 5 --bs_algo 10 --dataset FakeNews',
                          'python3 run_algo.py --dataset_size 0 --algo CATGAN --split 1 --nb_epoch_algo 15 --bs_algo 5 --dataset Irony ',
                          'python3 run_algo.py --dataset_size 0 --algo CATGAN --split 1 --nb_epoch_algo 5 --bs_algo 10 --dataset IronyB ']
                     },
          'CBERT': {100:['python3 run_algo.py --algo CBERT --split 1 --dataset_size 100 --bs_algo 32',
                         'python3 run_algo.py --algo CBERT --split 1 --dataset_size 100 --bs_algo 32 --dataset TREC6',
                         'python3 run_algo.py --algo CBERT --split 1 --dataset_size 100 --bs_algo 32 --dataset FakeNews',
                         'python3 run_algo.py --algo CBERT --split 1 --dataset_size 100 --bs_algo 32 --dataset Irony',
                         'python3 run_algo.py --algo CBERT --split 1 --dataset_size 100 --bs_algo 32 --dataset IronyB'],
                    500:['python3 run_algo.py --algo CBERT --split 1 --dataset_size 500 --bs_algo 32',
                         'python3 run_algo.py --algo CBERT --split 1 --dataset_size 500 --bs_algo 32 --dataset TREC6',
                         'python3 run_algo.py --algo CBERT --split 1 --dataset_size 500 --bs_algo 32 --dataset FakeNews',
                         'python3 run_algo.py --algo CBERT --split 1 --dataset_size 500 --bs_algo 32 --dataset Irony',
                         'python3 run_algo.py --algo CBERT --split 1 --dataset_size 500 --bs_algo 32 --dataset IronyB'],
                    1000:['python3 run_algo.py --algo CBERT --split 1 --dataset_size 1000 --bs_algo 32',
                         'python3 run_algo.py --algo CBERT --split 1 --dataset_size 1000 --bs_algo 32 --dataset TREC6',
                         'python3 run_algo.py --algo CBERT --split 1 --dataset_size 1000 --bs_algo 32 --dataset FakeNews',
                         'python3 run_algo.py --algo CBERT --split 1 --dataset_size 1000 --bs_algo 32 --dataset Irony',
                         'python3 run_algo.py --algo CBERT --split 1 --dataset_size 1000 --bs_algo 32 --dataset IronyB'],
                    0: ['python3 run_algo.py --algo CBERT --split 1 --dataset_size 0 --bs_algo 32',
                         'python3 run_algo.py --algo CBERT --split 1 --dataset_size 0 --bs_algo 32 --dataset TREC6',
                         'python3 run_algo.py --algo CBERT --split 1 --dataset_size 0 --bs_algo 32 --dataset FakeNews',
                         'python3 run_algo.py --algo CBERT --split 1 --dataset_size 0 --bs_algo 32 --dataset Irony',
                         'python3 run_algo.py --algo CBERT --split 1 --dataset_size 0 --bs_algo 32 --dataset IronyB']
                    },
          'GPT': {100:['python3 run_algo.py --algo GPT --split 1 --dataset_size 100 --nb_epoch_algo 20',
                       'python3 run_algo.py --algo GPT --split 1 --dataset_size 100 --nb_epoch_algo 20 --dataset TREC6',
                       'python3 run_algo.py --algo GPT --split 1 --dataset_size 100 --nb_epoch_algo 20 --dataset FakeNews',
                       'python3 run_algo.py --algo GPT --split 1 --dataset_size 100 --nb_epoch_algo 20 --dataset Irony',
                       'python3 run_algo.py --algo GPT --split 1 --dataset_size 100 --nb_epoch_algo 20 --dataset IronyB'],
                  500:['python3 run_algo.py --algo GPT --split 1 --dataset_size 500 --nb_epoch_algo 20',
                       'python3 run_algo.py --algo GPT --split 1 --dataset_size 500 --nb_epoch_algo 20 --dataset TREC6',
                       'python3 run_algo.py --algo GPT --split 1 --dataset_size 500 --nb_epoch_algo 20 --dataset FakeNews',
                       'python3 run_algo.py --algo GPT --split 1 --dataset_size 500 --nb_epoch_algo 20 --dataset Irony',
                       'python3 run_algo.py --algo GPT --split 1 --dataset_size 500 --nb_epoch_algo 20 --dataset IronyB'],
                  1000:['python3 run_algo.py --algo GPT --split 1 --dataset_size 1000 --nb_epoch_algo 20',
                       'python3 run_algo.py --algo GPT --split 1 --dataset_size 1000 --nb_epoch_algo 20 --dataset TREC6',
                       'python3 run_algo.py --algo GPT --split 1 --dataset_size 1000 --nb_epoch_algo 20 --dataset FakeNews',
                       'python3 run_algo.py --algo GPT --split 1 --dataset_size 1000 --nb_epoch_algo 20 --dataset Irony',
                       'python3 run_algo.py --algo GPT --split 1 --dataset_size 1000 --nb_epoch_algo 20 --dataset IronyB'],
                  0:['python3 run_algo.py --algo GPT --split 1 --dataset_size 0 --nb_epoch_algo 20',
                       'python3 run_algo.py --algo GPT --split 1 --dataset_size 0 --nb_epoch_algo 20 --dataset TREC6',
                       'python3 run_algo.py --algo GPT --split 1 --dataset_size 0 --nb_epoch_algo 20 --dataset FakeNews',
                       'python3 run_algo.py --algo GPT --split 1 --dataset_size 0 --nb_epoch_algo 20 --dataset Irony',
                       'python3 run_algo.py --algo GPT --split 1 --dataset_size 0 --nb_epoch_algo 20 --dataset IronyB']

                  }
          }
metrics=[1,3,1,1,3]
names=["SST2", "TREC6", "FakeNews", "Irony", "IronyB"]

def extractResult(results, dataset, classifier):
    # print(results)
    if classifier in ['xgboost', 'bert']:
        return results[metrics[dataset]]*100
    elif classifier=='dan':
        return max(results[metrics[dataset]])*100
#

if __name__=="__main__":
    #Average over dataset
    average_over_dataset= {"SST2":{"xgboost":[0,0], "dan":[0,0], "bert":[0,0]},
                           "TREC6":{"xgboost":[0,0], "dan":[0,0], "bert":[0,0]},
                           "FakeNews":{"xgboost":[0,0], "dan":[0,0], "bert":[0,0]},
                           "Irony":{"xgboost":[0,0], "dan":[0,0], "bert":[0,0]},
                           "IronyB":{"xgboost":[0,0], "dan":[0,0], "bert":[0,0]}}

    average_over_algorithms = {'xgboost':{0:[], 100:[], 500:[], 1000:[]},
                               'dan':{0:[], 100:[], 500:[], 1000:[]},
                               'bert':{0:[], 100:[], 500:[], 1000:[]},}

    variance_over_algorithms = {'xgboost': [],
                               'dan': [],
                               'bert': [], }

    output_full_results=False


    #SST-2 table
    print("")

    for algo, splitCommands in commands.items():

        average_over_classifier = {'xgboost': [], 'dan': [], 'bert': []}
        results_to_print=np.zeros((3, 5))
        baseline_to_print=np.zeros((3, 5))
        pvalue_to_print=np.zeros((3, 5))
        underlined=np.zeros((3, 5))
        for index_j, split in enumerate([100, 500, 1000, 0]):
            if index_j==3:
                index_j=4
            for index_i, classifier in enumerate(['xgboost', 'dan', 'bert']):
                baseline = np.zeros((30, 5))
                augmented = np.zeros((30, 5))
                for dataset, command in enumerate(splitCommands[split]):
                    # print(classifier, dataset)
                    # if classifier !="bert":
                    #     continue
                    # if dataset != 1:
                    #     continue
                    command = command + f" --classifier {classifier} --get_all_results"
                    # print(command)
                    process = subprocess.Popen(command.split(), stdout=open('tefdasfdasfdsadsfa', 'w'))
                    output, error = run_external_process(process)
                    results = json.load(open("temp.json", "r"))
                    aug = results['augmented']
                    bas = results['baseline']
                    # print(aug)
                    for i in range(30):
                        augmented[i, dataset] = extractResult(aug[str(i)], dataset, classifier)# aug[str(i)][metrics[dataset]] * 100
                        baseline[i, dataset] = extractResult(bas[str(i)], dataset, classifier)
                    if output_full_results:
                        results_temp=augmented[:, dataset]
                        baseline_temp=baseline[:, dataset]
                        ttest=ttest_rel(baseline_temp, results_temp)
                        print(names[dataset], algo, classifier, split, "results:", np.mean(results_temp), "baseline:", np.mean(baseline_temp), ttest)
                    # fds
                # print(augmented, baseline)
                # print(baseline)
                # fds
                if split!=0:
                    average_over_classifier[classifier].append(np.mean(augmented))
                    average_results = np.mean(augmented, axis=0)
                    baseline_results = np.mean(baseline, axis=0)
                    for i in range(5):
                        average_over_dataset[names[i]][classifier][0]+=baseline_results[i]
                        average_over_dataset[names[i]][classifier][1]+=average_results[i]
                # TODO HERE
                # continue
                average_over_algorithms[classifier][split].append(np.mean(augmented))
                variance_over_algorithms[classifier].extend(list(np.std(augmented-baseline, axis=0)))
                # print(variance_over_algorithms)
                # print(f"algo {algo} split {split} classifier {classifier} result {np.mean(augmented)} baseline {np.mean(baseline)} p-value {mult_paired_t_test(baseline, augmented)}")
                results_to_print[index_i, index_j]=np.mean(augmented)
                baseline_to_print[index_i, index_j]=np.mean(baseline)
                pvalue_to_print[index_i, index_j]=mult_paired_t_test(baseline, augmented)
                if np.mean(augmented)> np.mean(baseline):
                    underlined[index_i, index_j]=1
            if output_full_results:
                print('------')

        def get_string(results_to_print, underlined, pvalues, i, j, baseline=False, average=False):
            r=round(results_to_print[i, j], 1)
            if baseline:
                return str(r)
            if underlined[i,j]==1:
                string=f"\\underline{{{r}}}"
            else:
                string=str(r)
            pv=pvalues[i,j]
            if pv<0.05 and not average:
                st="**"
            elif pv<0.1 and not average:
                st="*"
            else:
                st=""
            return f"{string}{st}"
        # print(f"Average values: xgboost: {sum(average_over_classifier['xgboost'])/3}, dan: {sum(average_over_classifier['dan'])/3}, bert: {sum(average_over_classifier['bert'])/3}")
        results_to_print[0, 3]=sum(average_over_classifier['xgboost'])/3
        results_to_print[1, 3]=sum(average_over_classifier['dan'])/3
        results_to_print[2, 3]=sum(average_over_classifier['bert'])/3
        baseline_to_print[0, 3] = sum(baseline_to_print[0, :3]) / 3
        baseline_to_print[1, 3] = sum(baseline_to_print[1, :3]) / 3
        baseline_to_print[2, 3] = sum(baseline_to_print[2, :3]) / 3
        if algo=="EDA":
            # print(baseline_to_print)
            print(f"\multirow{{3}}{{*}}{{Baseline}}& xgboost & {get_string(baseline_to_print, underlined, pvalue_to_print, 0, 0, True)} & {get_string(baseline_to_print, underlined, pvalue_to_print, 0, 1, baseline=True)} & {get_string(baseline_to_print, underlined, pvalue_to_print, 0, 2, baseline=True)}  & {get_string(baseline_to_print, underlined, pvalue_to_print, 0, 3, baseline=True)}& {get_string(baseline_to_print, underlined, pvalue_to_print, 0, 4, baseline=True)}  \\\\  ")
            print(f"& FFN & {get_string(baseline_to_print, underlined, pvalue_to_print, 1, 0, baseline=True)} & {get_string(baseline_to_print, underlined, pvalue_to_print, 1, 1, baseline=True)} & {get_string(baseline_to_print, underlined, pvalue_to_print, 1, 2, baseline=True)}  & {get_string(baseline_to_print, underlined, pvalue_to_print, 1, 3, baseline=True)}& {get_string(baseline_to_print, underlined, pvalue_to_print, 1, 4, baseline=True)}  \\\\ ")
            print(f"& BERT & {get_string(baseline_to_print, underlined, pvalue_to_print, 2, 0, baseline=True)} & {get_string(baseline_to_print, underlined, pvalue_to_print, 2, 1, baseline=True)} & {get_string(baseline_to_print, underlined, pvalue_to_print, 2, 2, baseline=True)}  & {get_string(baseline_to_print, underlined, pvalue_to_print, 2, 3, baseline=True)}& {get_string(baseline_to_print, underlined, pvalue_to_print, 2, 4, baseline=True)}  \\\\ \\hline ")

        for index_of_average in range(3):
            if results_to_print[index_of_average, 3]>baseline_to_print[index_of_average, 3]:
                underlined[index_of_average, 3]=1

        if algo=="VAE_EncDec":
            algoname=f"\\vaeed"
        else:
            algoname=algo
        print(f"\multirow{{3}}{{*}}{{{algoname}}}& xgboost & {get_string(results_to_print, underlined, pvalue_to_print, 0,0)} & {get_string(results_to_print, underlined, pvalue_to_print, 0,1)} & {get_string(results_to_print, underlined, pvalue_to_print, 0,2)}  & {get_string(results_to_print, underlined, pvalue_to_print, 0,3, average=True)}& {get_string(results_to_print, underlined, pvalue_to_print, 0,4)}  \\\\  ")
        print(f"& FFN & {get_string(results_to_print, underlined, pvalue_to_print, 1,0)} & {get_string(results_to_print, underlined, pvalue_to_print, 1,1)} & {get_string(results_to_print, underlined, pvalue_to_print, 1,2)}  & {get_string(results_to_print, underlined, pvalue_to_print, 1,3, average=True)}& {get_string(results_to_print, underlined, pvalue_to_print, 1,4)}  \\\\ ")
        print(f"& BERT & {get_string(results_to_print, underlined, pvalue_to_print, 2,0)} & {get_string(results_to_print, underlined, pvalue_to_print, 2,1)} & {get_string(results_to_print, underlined, pvalue_to_print, 2,2)}  & {get_string(results_to_print, underlined, pvalue_to_print, 2,3, average=True)}& {get_string(results_to_print, underlined, pvalue_to_print, 2,4)}  \\\\ \\hline ")
        # & FFN  & 51.9 & \underline{58.1}** & \underline{61.7}** & \textbf{\underline{57.2}} &\underline{69.0}**\\
        # & bert & \underline{51.0}** & \underline{64.9}** & \underline{73.2}** & \underline{63.0} & \underline{80.3}**\\ \hline \hline")
        # print(f'========={algo}===========')
        # break
    # print(average_over_algorithms)
    print(f"\multirow{{3}}{{*}}{{Average}}& xgboost & {round(np.mean(average_over_algorithms['xgboost'][100]), 1)} & {round(np.mean(average_over_algorithms['xgboost'][500]), 1)} & {round(np.mean(average_over_algorithms['xgboost'][1000]), 1)}  & {round((np.mean(average_over_algorithms['xgboost'][1000])+np.mean(average_over_algorithms['xgboost'][500])+np.mean(average_over_algorithms['xgboost'][100]))/3, 1)} & {round(np.mean(average_over_algorithms['xgboost'][0]), 1)}  \\\\  ")
    print(f"& FFN & {round(np.mean(average_over_algorithms['xgboost'][100]), 1)} & {round(np.mean(average_over_algorithms['dan'][500]), 1)} & {round(np.mean(average_over_algorithms['dan'][1000]), 1)}  & {round((np.mean(average_over_algorithms['dan'][1000])+np.mean(average_over_algorithms['dan'][500])+np.mean(average_over_algorithms['dan'][100]))/3, 1)} & {round(np.mean(average_over_algorithms['dan'][0]), 1)}  \\\\ ")
    print(f"& BERT  & {round(np.mean(average_over_algorithms['bert'][100]), 1)} & {round(np.mean(average_over_algorithms['bert'][500]), 1)} & {round(np.mean(average_over_algorithms['bert'][1000]), 1)}  & {round((np.mean(average_over_algorithms['bert'][1000])+np.mean(average_over_algorithms['bert'][500])+np.mean(average_over_algorithms['bert'][100]))/3, 1)} & {round(np.mean(average_over_algorithms['bert'][0]), 1)}  \\\\ \\hline ")

    print("===========Average over datasets=============")

    #Normalizater=num_dataset*num_split_considered(3)
    normalizater=3*9

    tableBaseline=np.zeros((3, 5))
    tableAug=np.zeros((3, 5))

    #Normalizing
    for i, (dataset, classifiers) in enumerate(average_over_dataset.items()):
        print(f'----{dataset}----')
        for j, (classifier, results) in enumerate(classifiers.items()):
            tableBaseline[j, i]=results[0]/normalizater
            tableAug[j, i]=results[1]/normalizater
            print(f"{classifier} | baseline: {results[0]/normalizater}, augmented: {results[1]/normalizater}")

    def boldResult(tableBase, tableAugment, i, j):
        stringBas=tableBase[i,j]
        stringAug=tableAug[i,j]
        if stringBas>stringAug:
            return f"\\textbf{{{round(stringBas, 1)}}}/{round(stringAug, 1)}"
        else:
            return f"{round(stringBas, 1)}/\\textbf{{{round(stringAug, 1)}}}"

    print(f"xgboost & {round(tableBaseline[0, 0], 1)}/{round(tableAug[0,0], 1)} & {round(tableBaseline[0, 1], 1)}/{round(tableAug[0,1], 1)} & {round(tableBaseline[0, 2], 1)}/{round(tableAug[0,2], 1)} &{round(tableBaseline[0, 3], 1)}/{round(tableAug[0,3], 1)} & {round(tableBaseline[0, 4], 1)}/{round(tableAug[0,4], 1)} ")
    print(f"FFN & {round(tableBaseline[1, 0], 1)}/{round(tableAug[1, 0], 1)} & {round(tableBaseline[1, 1], 1)}/{round(tableAug[1, 1], 1)} & {round(tableBaseline[1, 2], 1)}/{round(tableAug[1, 2], 1)} &{round(tableBaseline[1, 3], 1)}/{round(tableAug[1, 3], 1)} & {round(tableBaseline[1, 4], 1)}/{round(tableAug[1, 4], 1)} ")
    print(f"xgboost & {round(tableBaseline[2, 0], 1)}/{round(tableAug[2, 0], 1)} & {round(tableBaseline[2, 1], 1)}/{round(tableAug[2, 1], 1)} & {round(tableBaseline[2, 2], 1)}/{round(tableAug[2, 2], 1)} &{round(tableBaseline[2, 3], 1)}/{round(tableAug[2, 3], 1)} & {round(tableBaseline[2, 4], 1)}/{round(tableAug[2, 4], 1)} ")
    # print("===========Average over everything==========")
    # print(average_over_algorithms)
    # for classifier, splits in average_over_algorithms.items():
    #     for split, results in splits.items():
    #         # print(results)
    #         # print(np.mean(results))
    #         average_over_algorithms[classifier][split]=np.mean(results)
    # print(average_over_algorithms)

    # print("===========Variance over everything==========")
    # print(variance_over_algorithms)
    # for classifier, results in variance_over_algorithms.items():
    #     print(classifier, np.mean(results))
    # print(average_over_algorithms)
    # for dataset, command in enumerate(GPT_100):
    #     command=command+" --classifier xgboost --get_all_results"
    #     process = subprocess.Popen(command.split(), stdout='temp')
    #     output, error = run_external_process(process)
    #     results=json.load(open("temp.json", "r"))
    #     aug=results['augmented']
    #     bas=results['baseline']
    #     for i in range(30):
    #         augmented[i, dataset]=aug[str(i)][metrics[dataset]]*100
    #         baseline[i, dataset]=bas[str(i)][metrics[dataset]]*100
    # print(augmented)
    # print(baseline)




    print("DONT FORGET TO CHANGE THE ABSTRACT, UNDELINE RESULT AND BOLD TABLE 2")
    print(mult_paired_t_test(baseline, augmented))