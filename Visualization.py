"""Define function for visualisation"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from transformers import AdamW, AutoTokenizer
# from torch.utils.data import DataLoader
import math, random
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from run_glue_DAN import DAN
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from run_glue_DAN import *

class CustomBERTModel(nn.Module):
    def __init__(self):
          super(CustomBERTModel, self).__init__()
          self.bert = BertModel.from_pretrained("bert-base-uncased")
          ### New layers:
          self.linear1 = nn.Linear(768, 1)

    def forward(self, ids, mask):
          sequence_output, pooled_output = self.bert(ids,attention_mask=mask)

          # sequence_output has the following shape: (batch_size, sequence_length, 768)
          linear1_output = self.linear1(sequence_output[:,0,:].view(-1,768)) ## extract the 1st token's embeddings

          # linear2_output = self.linear2(linear2_output)

          return linear1_output, sequence_output[:,0,:].view(-1,768)



def tsne_visualize_bert_linear(argdict):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    colors = ['red', 'blue', 'green', 'black', 'purple', 'darkorange']
    markers = ['.'] * len(argdict['categories']) + ['x'] * len(argdict['categories'])
    colors = colors[:len(argdict['categories'])] * 2
    labels_name = argdict['categories']
    labels_name = [f"{label}_original" for label in labels_name] + [f"{label}_generated" for label in labels_name]
    print(markers, colors)
    fig = plt.figure()
    SelectedData = pd.read_csv(
        f"{argdict['pathDataAdd']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}/train.tsv", sep='\t')

    # We train a bert to obtain the last layer representation
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = CustomBERTModel()
    # for param in self.model.base_model.parameters():
    #     param.requires_grad = False
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    # Selected Sentences
    all_sentences = []
    sentence_en_ordre = []
    labels_en_ordre = []
    all_labels = []
    lengths = []
    for i, cat in enumerate(argdict['categories']):
        # path = f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}/{cat}.txt"
        file = list(SelectedData[SelectedData['label'] == i]['sentence'])
        # print(file)
        all_sentences.extend(file)
        sentence_en_ordre.extend(file)
        lengths.append(len(file))
        all_labels.extend([i] * len(file))
        labels_en_ordre.extend([i] * len(file))

    # Train the model
    bs = 64
    criterion=nn.BCELoss()
    for j in range(5):
    # for j in range(0):
        c = list(zip(all_sentences, all_labels))
        random.shuffle(c)
        all_sentences, all_labels = zip(*c)
        all_sentences = list(all_sentences)
        all_labels = list(all_labels)
        pred = torch.zeros(len(all_labels))
        index = 0
        for i in range(math.ceil(len(all_sentences) / bs)):
            sent = all_sentences[index:index + bs]
            labels = all_labels[index:index + bs]
            # print(sent, labels)
            optimizer.zero_grad()
            encoding = tokenizer(sent, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            # print(encoding)
            # labels = batch['label']
            # print(input_ids)
            # print(labels)
            labels = torch.Tensor(labels).type(torch.LongTensor)
            outputs, last_layer = model(input_ids, mask=attention_mask)
            outputs=torch.clip(outputs, 0, 1)
            # print(labels)
            # print(outputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            # print(loss)
            # print(outputs)
            loss.backward()
            optimizer.step()

            # results = torch.argmax(torch.log_softmax(outputs[1], dim=1), dim=1)
            # print(outputs)
            results=torch.where(outputs>0.5, 1, 0).squeeze(1)
            # print(results)
            pred[index:index + bs] = results

            index += bs
        print(f"Epoch {j} Train Accuracy {accuracy_score(all_labels, pred)}")
    # all_sentences=[]
    # lengths=[]
    for i, cat in enumerate(argdict['categories']):
        path = f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}/{cat}.txt"
        file = open(path, 'r').readlines()[:50]
        all_sentences.extend(file)
        # print(len(file))
        sentence_en_ordre.extend(file)
        lengths.append(len(file))
        all_labels.extend([i] * len(file))
        labels_en_ordre.extend([i] * len(file))
    # print(all_sentences)

    # One more time to get the outputs
    # embeds=torch.zeros(len(all_labels), outputs[2][-1].shape[1], 768)
    # print(outputs[2][-1].shape)
    index = 0
    # for i in range(math.ceil(len(all_sentences) / bs)):
    model.eval()
    with torch.no_grad():
        sent = sentence_en_ordre
        labels = labels_en_ordre
        # print(sent, labels)
        encoding = tokenizer(sent, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        # print(encoding)
        # labels = batch['label']
        # print(input_ids)
        # print(labels)
        labels = torch.Tensor(labels).type(torch.LongTensor)
        outputs, last_layer = model(input_ids, mask=attention_mask)
        # embeds[index:index+bs]=outputs[2][-1]

    # outputs = outputs[2][-1].view(len(all_labels), -1)
    # print(embeds.shape)
    tsne = TSNE(n_components=2)

    features = tsne.fit_transform(last_layer)
    tot = 0
    # print(tsne_obj)
    # print(len(tsne_obj))
    # print(sentence_en_ordre)
    # print(labels_en_ordre)
    labels = argdict['categories']
    labels = [f"{label}_original" for label in labels] + [f"{label}_generated" for label in labels]
    print(labels_name)
    for i, ll in enumerate(lengths):
        plt.scatter(x=features[tot:tot + ll, 0], y=features[tot:tot + ll, 1], c=colors[i], label=labels_name[i],
                    marker=markers[i])
        tot += ll

    plt.legend()
    # plt.xlabel('Epochs')
    # plt.ylabel('Step L2 Norm')
    # plt.title(
    #     f't-sne visualization of the BERT features \n for the {argdict["dataset"]} dataset and {argdict["algo"]}')

    plt.savefig(f"Graphes/TSNE_{argdict['dataset']}_{argdict['algo']}_bert.png")

class danDS(Dataset):

    def __init__(self, vocab, sentences, labels, tokenizer, max_len=64):
        super().__init__()
        self.data={}
        self.max_len=max_len
        self.pad_id=vocab(['<pad>'])
        self.vocab=vocab
        self.pad_idx=self.vocab['<pad>']
        for i, (sentence, label) in enumerate(zip(sentences, labels)):
            self.data[i]={'input':vocab(tokenizer.tokenize(sentence)), 'label':label}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        input=self.data[item]['input'][:self.max_len]
        label=self.data[item]['label']
        input.extend([self.pad_idx] * (self.max_len - len(input)))

        return{
            'input':np.asarray(input),
            'label':label
        }


def tsne_visualize_dan_linear(argdict):
    colors = ['red', 'blue', 'green', 'black', 'purple', 'darkorange']
    markers = ['.'] * len(argdict['categories']) + ['x'] * len(argdict['categories'])
    colors = colors[:len(argdict['categories'])] * 2
    labels_name = argdict['categories']
    labels_name = [f"{label}_original" for label in labels_name] + [f"{label}_generated" for label in labels_name]
    print(markers, colors)
    fig = plt.figure()
    SelectedData = pd.read_csv(
        f"{argdict['pathDataAdd']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}/train.tsv", sep='\t')

    # We train a bert to obtain the last layer representation
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model=DAN(d_out=1).cuda()
    # for param in self.model.base_model.parameters():
    #     param.requires_grad = False
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Selected Sentences
    all_sentences = []
    sentence_en_ordre = []
    sentences_selected = []
    labels_selected = []
    labels_en_ordre = []
    all_labels = []
    lengths = []
    for i, cat in enumerate(argdict['categories']):
        # path = f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}/{cat}.txt"
        file = list(SelectedData[SelectedData['label'] == i]['sentence'])
        # print(file)
        all_sentences.extend(file)
        sentence_en_ordre.extend(file)
        sentences_selected.extend(file)
        labels_selected.extend([i] * len(file))
        lengths.append(len(file))
        all_labels.extend([i] * len(file))
        labels_en_ordre.extend([i] * len(file))


    nb_add = int(argdict['dataset_size'] / len(argdict['categories']))
    for i, cat in enumerate(argdict['categories']):
        path = f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}/{cat}.txt"
        file = open(path, 'r').readlines()[:nb_add]
        all_sentences.extend(file)
        all_labels.extend([i] * len(file))
        lengths.append(len(file))

    def iterator(sentences, tokenizer):
        for ss in sentences:
            yield tokenizer.tokenize(ss)


    #buildVocab
    from nltk.tokenize import TweetTokenizer
    from torchtext.vocab import build_vocab_from_iterator
    TT=TweetTokenizer()
    vocab=build_vocab_from_iterator(iterator(all_sentences, TT), specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])
    # print(vocab.get_itos())
    # vocab.get
    #Build dataset for the regular model
    ds=danDS(vocab, sentences_selected, labels_selected, TT)


    # for i, batch in enumerate(iterator):
    #     print(batch)

    # Train the model
    bs = 64
    criterion=nn.BCELoss()
    for j in range(3):
        #
        iteratorData = DataLoader(
            dataset=ds,
            batch_size=bs,
            shuffle=True,
            pin_memory=torch.cuda.is_available()
        )

        pred = torch.zeros(len(labels_selected))
        labels = torch.zeros(len(labels_selected))
        index = 0
        for batch in iteratorData:
            input=batch['input'].cuda()
            label=batch['label'].cuda()
            optimizer.zero_grad()
            output=model((input, 2))
            # print(output)
            output=torch.clip(output, 0, 1)
            # print(output)
            loss = criterion(output, label.unsqueeze(1).float())
            loss.backward()
            optimizer.step()


            results = torch.where(output > 0.5, 1, 0).squeeze(1)
            # print('---')
            pred[index:index + bs] = results
            labels[index:index + bs] = label

            index += bs
        print(f"Epoch {j} Train Accuracy {accuracy_score(labels, pred)}")

    #model is trained, now let's encode all the data
    model.eval()
    ds = danDS(vocab, all_sentences, all_labels, TT)

    # for i, batch in enumerate(iterator):
    #     print(batch)

    # Train the model
    iteratorData = DataLoader(
        dataset=ds,
        batch_size=len(all_labels),
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )
    with torch.no_grad():
        for batch in iteratorData:

            last_layer=model.return_last_layer(batch['input'].cuda()).cpu()

    tsne = TSNE(n_components=2)

    features = tsne.fit_transform(last_layer)
    tot = 0
    # print(tsne_obj)
    # print(len(tsne_obj))
    # print(sentence_en_ordre)
    # print(labels_en_ordre)
    labels = argdict['categories']
    labels = [f"{label}_original" for label in labels] + [f"{label}_generated" for label in labels]
    print(labels_name)
    for i, ll in enumerate(lengths):
        plt.scatter(x=features[tot:tot + ll, 0], y=features[tot:tot + ll, 1], c=colors[i], label=labels_name[i],
                    marker=markers[i])
        tot += ll

    plt.legend()
    # plt.xlabel('Epochs')
    # plt.ylabel('Step L2 Norm')
    # plt.title(
    #     f't-sne visualization of the FFN features \n for the {argdict["dataset"]} dataset and {argdict["algo"]}')

    plt.savefig(f"Graphes/TSNE_{argdict['dataset']}_{argdict['algo']}_dan.png")

def tsne_visualize(argdict):
    colors = ['red', 'blue', 'green', 'black', 'purple', 'darkorange']
    markers=['.']*len(argdict['categories'])+['x']*len(argdict['categories'])
    colors=colors[:len(argdict['categories'])]*2
    labels=argdict['categories']
    labels=[f"{label}_original" for label in labels]+[f"{label}_generated" for label in labels]
    print(markers, colors)
    fig = plt.figure()
    SelectedData= pd.read_csv(f"{argdict['pathDataAdd']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}/train.tsv", sep='\t')

    #Selected Sentences
    all_sentences = []
    sentences_selected=[]
    lengths = []
    all_words=[]
    for i, cat in enumerate(argdict['categories']):
        # path = f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}/{cat}.txt"
        file = list(SelectedData[SelectedData['label']==i]['sentence'])
        all_words.extend([words for sentences in file for words in sentences.split(' ')])
        all_sentences.extend(file)
        lengths.append(len(file))
        sentences_selected.extend(file)


    # all_sentences=[]
    # lengths=[]
    nb_add=int(argdict['dataset_size']/len(argdict['categories']))
    for cat in argdict['categories']:
        path = f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}/{cat}.txt"
        file=open(path, 'r').readlines()[:nb_add]
        all_words.extend([words for sentences in file for words in sentences.split(' ')])
        all_sentences.extend(file)
        lengths.append(len(file))
    # print(all_sentences)

    all_words=set(all_words)
    print(all_words)

    tfidf_transformer = TfidfVectorizer(vocabulary=all_words)
    tfidf_transformer.fit(sentences_selected)


    x_train_tf=tfidf_transformer.transform(all_sentences)
    tsne = TSNE(n_components=2)

    features = tsne.fit_transform(x_train_tf)
    tot = 0
    # print(tsne_obj)
    # print(len(tsne_obj))
    for i, ll in enumerate(lengths):
        plt.scatter(x=features[tot:tot + ll, 0], y=features[tot:tot + ll, 1], c=colors[i], label=labels[i], marker=markers[i])
        tot += ll

    plt.legend()
    # plt.xlabel('Epochs')
    # plt.ylabel('Step L2 Norm')
    # plt.title(f't-sne visualization of the TF-IDF features \n for the {argdict["dataset"]} dataset and {argdict["algo"]}')

    plt.savefig(f"Graphes/TSNE_{argdict['dataset']}_{argdict['algo']}.png")

def tsne_visualize_bert(argdict):
    colors = ['red', 'blue', 'green', 'black']
    markers=['.']*len(argdict['categories'])+['x']*len(argdict['categories'])
    colors=colors[:len(argdict['categories'])]*2
    labels_name=argdict['categories']
    labels_name=[f"{label}_original" for label in labels_name]+[f"{label}_generated" for label in labels_name]
    print(markers, colors)
    fig = plt.figure()
    SelectedData= pd.read_csv(f"{argdict['pathDataAdd']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}/train.tsv", sep='\t')

    #We train a bert to obtain the last layer representation
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',  output_hidden_states=True)
    # for param in self.model.base_model.parameters():
    #     param.requires_grad = False
    optimizer = AdamW(model.parameters(), lr=1e-5)

        #Selected Sentences
    all_sentences = []
    sentence_en_ordre=[]
    labels_en_ordre=[]
    all_labels= []
    lengths = []
    for i, cat in enumerate(argdict['categories']):
        # path = f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}/{cat}.txt"
        file = list(SelectedData[SelectedData['label']==i]['sentence'])
        # print(file)
        all_sentences.extend(file)
        sentence_en_ordre.extend(file)
        lengths.append(len(file))
        all_labels.extend([i]*len(file))
        labels_en_ordre.extend([i]*len(file))

    # Train the model
    bs = 64

    for j in range(5):
        c = list(zip(all_sentences, all_labels))
        random.shuffle(c)
        all_sentences, all_labels = zip(*c)
        all_sentences = list(all_sentences)
        all_labels = list(all_labels)
        pred = torch.zeros(len(all_labels))
        index = 0
        for i in range(math.ceil(len(all_sentences) / bs)):
            sent = all_sentences[index:index + bs]
            labels = all_labels[index:index + bs]
            # print(sent, labels)
            optimizer.zero_grad()
            encoding = tokenizer(sent, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            # print(encoding)
            # labels = batch['label']
            # print(input_ids)
            # print(labels)
            labels = torch.Tensor(labels).type(torch.LongTensor)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            # print(outputs)
            loss = outputs[0]
            # print(loss)
            # print(outputs)
            loss.backward()
            optimizer.step()

            results = torch.argmax(torch.log_softmax(outputs[1], dim=1), dim=1)
            pred[index:index + bs] = results

            index += bs
        print(f"Epoch {j} Train Accuracy {accuracy_score(all_labels, pred)}")


    # all_sentences=[]
    # lengths=[]
    for i, cat in enumerate(argdict['categories']):
        path = f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}/{cat}.txt"
        file=open(path, 'r').readlines()
        print(file)
        all_sentences.extend(file)
        sentence_en_ordre.extend(file)
        lengths.append(len(file))
        all_labels.extend([i] * len(file))
        labels_en_ordre.extend([i]*len(file))
    # print(all_sentences)

    #One more time to get the outputs
    # embeds=torch.zeros(len(all_labels), outputs[2][-1].shape[1], 768)
    # print(outputs[2][-1].shape)
    index = 0
    # for i in range(math.ceil(len(all_sentences) / bs)):
    model.eval()
    with torch.no_grad():
        sent = sentence_en_ordre
        labels = labels_en_ordre
        # print(sent, labels)
        encoding = tokenizer(sent, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        # print(encoding)
        # labels = batch['label']
        # print(input_ids)
        # print(labels)
        labels = torch.Tensor(labels).type(torch.LongTensor)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        # embeds[index:index+bs]=outputs[2][-1]

    outputs=outputs[2][-1].view(len(all_labels), -1)
    # print(embeds.shape)
    tsne = TSNE(n_components=2)

    features = tsne.fit_transform(outputs)
    tot = 0
    # print(tsne_obj)
    # print(len(tsne_obj))
    # print(sentence_en_ordre)
    # print(labels_en_ordre)
    labels = argdict['categories']
    labels = [f"{label}_original" for label in labels] + [f"{label}_generated" for label in labels]
    print(labels_name)
    for i, ll in enumerate(lengths):
        plt.scatter(x=features[tot:tot + ll, 0], y=features[tot:tot + ll, 1], c=colors[i], label=labels_name[i], marker=markers[i])
        tot += ll

    plt.legend()
    # plt.xlabel('Epochs')
    # plt.ylabel('Step L2 Norm')
    plt.title(f't-sne visualization of the TF-IDF features \n for the {argdict["dataset"]} dataset and {argdict["algo"]}')

    plt.savefig(f"test_bert.png")

def pca_visualize(argdict):
    colors = ['red', 'blue', 'green', 'black']
    markers=['.']*len(argdict['categories'])+['x']*len(argdict['categories'])
    colors=colors[:len(argdict['categories'])]*2
    labels=argdict['categories']
    labels=[f"{label}_original" for label in labels]+[f"{label}_generated" for label in labels]
    print(markers, colors)
    fig = plt.figure()
    SelectedData= pd.read_csv(f"{argdict['pathDataAdd']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}/train.tsv", sep='\t')

    #Selected Sentences
    all_sentences = []
    lengths_OG = []
    for i, cat in enumerate(argdict['categories']):
        # path = f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}/{cat}.txt"
        file = list(SelectedData[SelectedData['label']==i]['sentence'])
        # print(file)
        all_sentences.extend(file)
        lengths_OG.append(len(file))

    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    x_train_count = count_vect.fit_transform(all_sentences)
    x_train_tf = tfidf_transformer.fit_transform(x_train_count)

    pca = TruncatedSVD(n_components=2)
    features_OG = pca.fit_transform(x_train_tf)

    all_sentences=[]
    lengths=[]
    for cat in argdict['categories']:
        path = f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}/{cat}.txt"
        file=open(path, 'r').readlines()
        all_sentences.extend(file)
        lengths.append(len(file))
    # print(all_sentences)

    # count_vect = CountVectorizer()
    # tfidf_transformer = TfidfTransformer()
    x_train_count = count_vect.transform(all_sentences)
    x_train_tf = tfidf_transformer.transform(x_train_count)


    features_added = pca.transform(x_train_tf)
    tot = 0
    # print(tsne_obj)
    # print(len(tsne_obj))
    features=np.concatenate((features_OG, features_added), axis=0)#features_OG.extend(features_added)
    lengths_OG.extend(lengths)
    lengths=lengths_OG
    print(features, lengths)
    for i, ll in enumerate(lengths):
        print(i)
        plt.scatter(x=features[tot:tot + ll, 0], y=features[tot:tot + ll, 1], c=colors[i], label=labels[i], marker=markers[i])
        tot += ll

    plt.legend()
    # plt.xlabel('Epochs')
    # plt.ylabel('Step L2 Norm')
    plt.title(f'SVD visualization of the BERT features \n for the {argdict["dataset"]} dataset and {argdict["algo"]}')

    plt.savefig(f"test_svd.png")

def pca_visualize_bert(argdict):
    colors = ['red', 'blue', 'green', 'black']
    markers=['.']*len(argdict['categories'])+['x']*len(argdict['categories'])
    colors=colors[:len(argdict['categories'])]*2
    labels=argdict['categories']
    labels=[f"{label}_original" for label in labels]+[f"{label}_generated" for label in labels]
    print(markers, colors)
    fig = plt.figure()
    SelectedData= pd.read_csv(f"{argdict['pathDataAdd']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}/train.tsv", sep='\t')

    # We train a bert to obtain the last layer representation
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', output_hidden_states=True)
    # for param in self.model.base_model.parameters():
    #     param.requires_grad = False
    optimizer = AdamW(model.parameters(), lr=1e-5)

    #Selected Sentences
    all_sentences = []
    sentence_en_ordre=[]
    labels_en_ordre=[]
    lengths = []
    labels_OG = []
    for i, cat in enumerate(argdict['categories']):
        # path = f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}/{cat}.txt"
        file = list(SelectedData[SelectedData['label']==i]['sentence'])
        # print(file)
        all_sentences.extend(file)
        sentence_en_ordre.extend(file)
        lengths.append(len(file))
        labels_OG.extend([i] * len(file))
        labels_en_ordre.extend([i] * len(file))

    bs = 64

    for j in range(5):
        c = list(zip(all_sentences, labels_OG))
        random.shuffle(c)
        all_sentences, labels_OG = zip(*c)
        all_sentences = list(all_sentences)
        labels_OG = list(labels_OG)
        pred = torch.zeros(len(labels_OG))
        index = 0
        for i in range(math.ceil(len(all_sentences) / bs)):
            sent = all_sentences[index:index + bs]
            labels = labels_OG[index:index + bs]
            # print(sent, labels)
            optimizer.zero_grad()
            encoding = tokenizer(sent, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            # print(encoding)
            # labels = batch['label']
            # print(input_ids)
            # print(labels)
            labels = torch.Tensor(labels).type(torch.LongTensor)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            # print(outputs)
            loss = outputs[0]
            # print(loss)
            # print(outputs)
            loss.backward()
            optimizer.step()

            results = torch.argmax(torch.log_softmax(outputs[1], dim=1), dim=1)
            pred[index:index + bs] = results

            index += bs
        print(f"Epoch {j} Train Accuracy {accuracy_score(labels_OG, pred)}")

    pca = TruncatedSVD(n_components=2)


    for i, cat in enumerate(argdict['categories']):
        path = f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}/{cat}.txt"
        file=open(path, 'r').readlines()
        all_sentences.extend(file)
        sentence_en_ordre.extend(file)
        labels_en_ordre.extend([i] * len(file))
        lengths.append(len(file))
    # print(all_sentences)

        # One more time to get the outputs
        # embeds=torch.zeros(len(all_labels), outputs[2][-1].shape[1], 768)
        # print(outputs[2][-1].shape)
        index = 0
    # for i in range(math.ceil(len(all_sentences) / bs)):
    model.eval()
    with torch.no_grad():
        sent = sentence_en_ordre
        labels = labels_en_ordre
        # print(sent, labels)
        encoding = tokenizer(sent, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        # print(encoding)
        # labels = batch['label']
        # print(input_ids)
        # print(labels)
        labels = torch.Tensor(labels).type(torch.LongTensor)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        # embeds[index:index+bs]=outputs[2][-1]

    outputs = outputs[2][-1].view(len(labels_en_ordre), -1)

    outputs_OG=outputs[:lengths[0]+lengths[1]]
    pca.fit(outputs_OG)
    features=pca.transform(outputs)

    tot=0
    labels = argdict['categories']
    labels_name = [f"{label}_original" for label in labels] + [f"{label}_generated" for label in labels]
    print(features.shape)
    print(lengths)
    print(labels_name)
    for i, ll in enumerate(lengths):
        plt.scatter(x=features[tot:tot + ll, 0], y=features[tot:tot + ll, 1], c=colors[i], label=labels_name[i], marker=markers[i])
        tot += ll

    plt.legend()
    # plt.xlabel('Epochs')
    # plt.ylabel('Step L2 Norm')
    plt.title(f'SVD visualization of the TF-IDF features \n for the {argdict["dataset"]} dataset and {argdict["algo"]}')

    plt.savefig(f"test_svd_bert.png")