"""Lime analysis"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import pandas as pd
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
# from torch.utils.data import DataLoader
import math, random
import torch
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


def lime_xgboost(argdict):
    """Lime analysis with xgboost"""
    labels = argdict['categories']
    # labels = [f"{label}_selected" for label in labels] + [f"{label}_generated" for label in labels]
    SelectedData = pd.read_csv(f"{argdict['pathDataAdd']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}/train.tsv", sep='\t')
    # print(SelectedData)
    model_baseline = XGBClassifier()
    model_augmented = XGBClassifier()

    #Train on basline, get prediction and true result
    dfVal = pd.read_csv(f'{argdict["pathDataOG"]}/{argdict["dataset"]}/dev.tsv', sep='\t')
    # print(dfVal)

    tfidf_transformer_baseline = TfidfVectorizer()
    # x_train_count = count_vect.fit_transform(all_sentences)
    x_train_tf = tfidf_transformer_baseline.fit_transform(SelectedData['sentence'])
    model_baseline.fit(x_train_tf, SelectedData['label'])
    y_pred_train = model_baseline.predict(x_train_tf)
    y_pred_val = model_baseline.predict(tfidf_transformer_baseline.transform(dfVal['sentence']))
    # if task not in ["MNLI", "STS-B", "QNLI", "RTE", "WNLI"]:

    predictions_train = [round(value) for value in y_pred_train]
    predictions = [round(value) for value in y_pred_val]
    true_labels=dfVal['label']
    true_labels_train=SelectedData['label']
    print(f"Accuracy on the train set for the baseline: {accuracy_score(true_labels_train, predictions_train)}")
    print(f"Accuracy on the validation set for the baseline: {accuracy_score(true_labels, predictions)}")

    index=len(SelectedData)
    for i, cat in enumerate(argdict['categories']):
        path = f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}/{cat}.txt"
        file = open(path, 'r').readlines()
        for line in file:
            SelectedData.at[index, 'sentence']=line
            SelectedData.at[index, 'label']=i
            index+=1
    # print(all_sentences)
    # print(SelectedData)
    SelectedData=SelectedData.sample(frac=1)
    dfVal = pd.read_csv(f'{argdict["pathDataOG"]}/{argdict["dataset"]}/dev.tsv', sep='\t')

    tfidf_transformer_aug = TfidfVectorizer()
    # x_train_count = count_vect.fit_transform(all_sentences)
    x_train_tf = tfidf_transformer_aug.fit_transform(SelectedData['sentence'])
    model_augmented.fit(x_train_tf, SelectedData['label'])

    y_pred_val = model_augmented.predict(tfidf_transformer_aug.transform(dfVal['sentence']))
    # if task not in ["MNLI", "STS-B", "QNLI", "RTE", "WNLI"]:
    predictions_aug = [round(value) for value in y_pred_val]
    true_labels = dfVal['label']
    print(f"Accuracy on the validation set for the augmented dataset: {accuracy_score(true_labels, predictions_aug)}")

    c_baseline=make_pipeline(tfidf_transformer_baseline, model_baseline)
    explainer_baseline=LimeTextExplainer(class_names=labels)

    c_augmented = make_pipeline(tfidf_transformer_aug, model_augmented)
    explainer_augmented = LimeTextExplainer(class_names=labels)

    correct_baseline = [1 if pred == tl else 0 for (pred, tl) in zip(predictions, true_labels)]
    correct_aug = [1 if pred == tl else 0 for (pred, tl) in zip(predictions_aug, true_labels)]
    bad_to_correct = np.array([1 if (baseline == 0 and aug == 1) else 0 for (baseline, aug) in zip(correct_baseline, correct_aug)])
    correct_to_bad = np.array([1 if (baseline == 1 and aug == 0) else 0 for (baseline, aug) in zip(correct_baseline, correct_aug)])
    print("------------------")
    print("Selecting 5 examples that the labels was wrong and became right")
    examples=np.where(bad_to_correct==1)[0][:5]

    """TEST GRAPH HERE"""
    #We want
    dico={}
    dico_aug={}
    with torch.no_grad():
        for i, sentence in enumerate(dfVal['sentence']):
            # if i==10:
            #     break
            exp = explainer_baseline.explain_instance(sentence, c_baseline.predict_proba, num_features=3)
            exp_aug = explainer_augmented.explain_instance(sentence, c_augmented.predict_proba, num_features=3)
            # probs=c_baseline.predict_proba([sentence])[0]#[0, 1])
            for elem in exp.as_list():
                try:
                    dico[elem[0]].append(elem[1])
                except:
                    dico[elem[0]]=[elem[1]]
            for elem in exp_aug.as_list():
                try:
                    dico_aug[elem[0]].append(elem[1])
                except:
                    dico_aug[elem[0]] = [elem[1]]
            # explainer_augmented.explain_instance(sentence, c_augmented.predict_proba, num_features=3)
    # print(dico)
    # print(dico_aug)
    dico={key:value for key, value in dico.items() if len(value)>5}
    print(dico.keys())
    dico_aug={key:value for key, value in dico_aug.items() if len(value)>5}
    print(len(dico))
    print(len(dico_aug))
    key=list(dico.keys())
    values=list(dico.values())
    values=[np.mean(val) for val in values]

    for word_aug in dico_aug.keys():
        if word_aug not in key:
            key.append(word_aug)
            dico[word_aug]=[0]

    for word in key:
        if word not in list(dico_aug.keys()):
            dico_aug[word]=[0]

    y_baseline=[np.mean(dico[k]) for k in key]
    y_aug=[np.mean(dico_aug[k]) for k in key]

    print(y_baseline)
    print(values)
    data=[23,24,5]

    Fig=plt.figure()
    plt.xticks(range(len(key)), key, rotation='vertical')
    plt.ylabel("Correlation with the positive class")
    plt.bar(np.arange(len(key)) - 0.2, y_baseline, 0.4, color='red', label='baseline')
    plt.bar(np.arange(len(key)) + 0.2, y_aug, 0.4,  color='blue', label='augmented')
    plt.legend()
    plt.savefig(f'../Figures/LIME_{argdict["dataset"]}_{argdict["starting_size"]}_{argdict["split"]}_xgboost.png', bbox_inches='tight')
    fdas
    # print(examples)
    for idx in examples:
        # print(idx)
        print("BASELINE")
        print(dfVal['sentence'][idx])
        exp = explainer_baseline.explain_instance(dfVal['sentence'][idx], c_baseline.predict_proba, num_features=6)
        print('Document id: %d' % idx)
        print(f'Probability({labels[1]}) =', c_baseline.predict_proba([dfVal['sentence'][idx]])[0, 1])
        print('True class: %s' % labels[dfVal['label'][idx]])
        print(exp.as_list())
        print("\nAUGMENTED")
        print(dfVal['sentence'][idx])
        exp = explainer_augmented.explain_instance(dfVal['sentence'][idx], c_augmented.predict_proba, num_features=6)
        print('Document id: %d' % idx)
        print(f'Probability({labels[1]}) =', c_augmented.predict_proba([dfVal['sentence'][idx]])[0, 1])
        print('True class: %s' % labels[dfVal['label'][idx]])
        print(exp.as_list())
        print('*****************')

    print("Selecting 5 examples that the labels was correct and became wrong")
    examples = np.where(correct_to_bad == 1)[0][:5]
    for idx in examples:
        print("BASELINE")
        print(dfVal['sentence'][idx])
        exp = explainer_baseline.explain_instance(dfVal['sentence'][idx], c_baseline.predict_proba, num_features=6)
        print('Document id: %d' % idx)
        print(f'Probability({labels[1]}) =', c_baseline.predict_proba([dfVal['sentence'][idx]])[0, 1])
        print('True class: %s' % labels[dfVal['label'][idx]])
        print(exp.as_list())
        print("\nAUGMENTED")
        print(dfVal['sentence'][idx])
        exp = explainer_augmented.explain_instance(dfVal['sentence'][idx], c_augmented.predict_proba, num_features=6)
        print('Document id: %d' % idx)
        print(f'Probability({labels[1]}) =', c_augmented.predict_proba([dfVal['sentence'][idx]])[0, 1])
        print('True class: %s' % labels[dfVal['label'][idx]])
        print(exp.as_list())
        print('*****************')

def train_model(model, x, y):
    """Train the bert model"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    optimizer = AdamW(model.parameters(), lr=1e-5)
    bs = 64

    # print("WARNING WARNING NOT TRAINING THE MODEL PROPERLY")
    for j in range(20):
        c = list(zip(x, y))
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
            # print(labels)
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
    return model

def predict_label(model, x, y):
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with torch.no_grad():
        sent = list(x)
        labels = list(y)
        # print(sent, labels)
        encoding = tokenizer(sent, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        # print(encoding)
        # labels = batch['label']
        # print(input_ids)
        # print(labels)
        labels = torch.Tensor(labels).type(torch.LongTensor)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)[1]
        labels=torch.argmax(outputs, dim=1)
        return labels



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def lime_bert(argdict):
    """Lime analysis with xgboost"""
    labels = argdict['categories']
    # labels = [f"{label}_selected" for label in labels] + [f"{label}_generated" for label in labels]
    SelectedData = pd.read_csv(f"{argdict['pathDataAdd']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}/train.tsv", sep='\t')
    # print(SelectedData)
    # model_baseline = XGBClassifier()
    # model_augmented = XGBClassifier()

    # import time
    # t = 1000 * time.time()  # current time in milliseconds
    # set_seed(int(t) % 2 ** 32)
    set_seed(32)
    # Train on basline, get prediction and true result
    dfVal = pd.read_csv(f'{argdict["pathDataOG"]}/{argdict["dataset"]}/dev.tsv', sep='\t')
    # print(dfVal)
    model_baseline=BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels))
    model_baseline=train_model(model_baseline, SelectedData['sentence'], SelectedData['label'])
    # if task not in ["MNLI", "STS-B", "QNLI", "RTE", "WNLI"]:

    predictions_train=predict_label(model_baseline, SelectedData['sentence'], SelectedData['label'])
    predictions=predict_label(model_baseline, dfVal['sentence'], dfVal['label'])

    # predictions_train = [round(value) for value in y_pred_train]
    # predictions = [round(value) for value in y_pred_val]
    true_labels = dfVal['label']
    true_labels_train = SelectedData['label']
    print(f"Accuracy/F1 on the train set for the baseline: {accuracy_score(true_labels_train, predictions_train)} / {f1_score(true_labels_train, predictions_train, average='macro')}")
    print(f"Accuracy/F1 on the validation set for the baseline: {accuracy_score(true_labels, predictions)}/{f1_score(true_labels, predictions, average='macro')}")

    index = len(SelectedData)
    for i, cat in enumerate(argdict['categories']):
        path = f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}/{cat}.txt"
        file = open(path, 'r').readlines()
        for j, line in enumerate(file):
            if j==int(argdict['dataset_size']/len(argdict['categories'])):
                break
            SelectedData.at[index, 'sentence'] = line
            SelectedData.at[index, 'label'] = i
            index += 1
    # print(all_sentences)
    # print(SelectedData)
    SelectedData = SelectedData.sample(frac=1)
    dfVal = pd.read_csv(f'{argdict["pathDataOG"]}/{argdict["dataset"]}/dev.tsv', sep='\t')

    # x_train_count = count_vect.fit_transform(all_sentences)
    model_augmented = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels))
    model_augmented= train_model(model_augmented, SelectedData['sentence'], SelectedData['label'])

    predictions_aug=predict_label(model_augmented, dfVal['sentence'], dfVal['label'])
    # if task not in ["MNLI", "STS-B", "QNLI", "RTE", "WNLI"]:
    # predictions_aug = [round(value) for value in y_pred_val]
    true_labels = dfVal['label']
    print(f"Accuracy/f1 on the validation set for the augmented dataset: {accuracy_score(true_labels, predictions_aug)}/{f1_score(true_labels, predictions_aug, average='macro')}")

    def predict_baseline(texts):
        model_baseline.eval()
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        with torch.no_grad():
            sent = list(texts)
            labels = np.zeros(len(sent))
            # print(sent, labels)
            encoding = tokenizer(sent, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            # print(encoding)
            # labels = batch['label']
            # print(input_ids)
            # print(labels)
            labels = torch.Tensor(labels).type(torch.LongTensor)
            outputs = torch.softmax(model_baseline(input_ids, attention_mask=attention_mask, labels=labels)[1], dim=1)
            # print(outputs)
            return outputs

    def predict_aug(texts):
        model_augmented.eval()
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        with torch.no_grad():
            sent = list(texts)
            labels = np.zeros(len(sent))
            # print(sent, labels)
            encoding = tokenizer(sent, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            # print(encoding)
            # labels = batch['label']
            # print(input_ids)
            # print(labels)
            labels = torch.Tensor(labels).type(torch.LongTensor)
            outputs = torch.softmax(model_augmented(input_ids, attention_mask=attention_mask, labels=labels)[1], dim=1)
            return outputs

    explainer_baseline = LimeTextExplainer(class_names=labels)

    # c_augmented = make_pipeline(tfidf_transformer_aug, model_augmented)
    explainer_augmented = LimeTextExplainer(class_names=labels,)

    correct_baseline = [1 if pred == tl else 0 for (pred, tl) in zip(predictions, true_labels)]
    correct_aug = [1 if pred == tl else 0 for (pred, tl) in zip(predictions_aug, true_labels)]
    bad_to_correct = np.array([1 if (baseline == 0 and aug == 1) else 0 for (baseline, aug) in zip(correct_baseline, correct_aug)])
    correct_to_bad = np.array([1 if (baseline == 1 and aug == 0) else 0 for (baseline, aug) in zip(correct_baseline, correct_aug)])

    print(sum(correct_baseline))
    print(sum(correct_aug))
    print(sum(bad_to_correct))
    print(sum(correct_to_bad))

    # print(predictions_aug)
    # print(true_labels)
    # print(correct_baseline)
    # print(correct_aug)

    """Graphe for 1 example"""
    examples = np.where(bad_to_correct == 1)[0][:5]
    print(examples[:10])
    # idx=examples[0]
    idx=15



    print("BASELINE")
    print(dfVal['sentence'][idx])
    exp = explainer_baseline.explain_instance(dfVal['sentence'][idx], predict_baseline, num_features=10, labels=[0,1,2,3,4,5])
    exp_aug = explainer_augmented.explain_instance(dfVal['sentence'][idx], predict_aug,
                                              num_features=10, labels=[0,1,2,3,4,5])
    print('Document id: %d' % idx)
    pred_baseline = predict_baseline([dfVal['sentence'][idx]]).squeeze(0)
    print(f'Probability({labels[torch.argmax(pred_baseline)]}) =', torch.max(pred_baseline))
    print('True class: %s' % labels[dfVal['label'][idx]])
    words_label=dfVal['sentence'][idx].split(' ')
    words_from_lime=[a[0] for a in exp.as_list(label=0)]
    true_labels_words=[]
    for ww in words_label:
        if ww in words_from_lime:
            true_labels_words.append(ww)
    lists_values=[[] for _ in range(len(labels))]
    lists_values_aug=[[] for _ in range(len(labels))]
    for i, word in enumerate(true_labels_words):
        #For each word, for each label, get associated value
        for j in range(len(labels)):
            value=[a[1] for a in exp.as_list(label=j) if a[0]==word]
            lists_values[j].extend(value)
            value = [a[1] for a in exp_aug.as_list(label=j) if a[0] == word]
            lists_values_aug[j].extend(value)
    fig, ax = plt.subplots()
    xaxis=np.arange(len(true_labels_words))
    colors=['#310A31', '#847996', '#88B7B5', '#A7CAB1', '#931F1D', '#1A5E63']
    print(lists_values)
    for i in range(len(labels)):
        ax.bar(xaxis-0.2, lists_values[i], 0.4, label="Pre-augmentation", color=colors[i%len(colors)])
    plt.xticks(xaxis, true_labels_words)
    for i in range(len(labels)):
        ax.bar(xaxis+0.2, lists_values_aug[i], 0.4, label="Post-augmentation", color=colors[i%len(colors)])
    ax.legend(argdict['categories'])
    print(lists_values)
    # for i i


    plt.savefig("testFig.png")

    # print(exp.as_list(label=1))
    # print(exp.as_list(label=2))
    # print(exp.as_list(label=3))
    # print(exp.as_list(label=4))
    # print(exp.available_labels())
    # print("\nAUGMENTED")
    # print(dfVal['sentence'][idx])
    # exp = explainer_augmented.explain_instance(dfVal['sentence'][idx], predict_aug,
    #                                            num_features=6)
    # print('Document id: %d' % idx)
    # pred_aug = predict_aug([dfVal['sentence'][idx]]).squeeze(0)
    # print(f'Probability({labels[torch.argmax(pred_aug)]}) =', torch.max(pred_aug))
    # print('True class: %s' % labels[dfVal['label'][idx]])
    # print(exp.as_list())
    # print('*****************')

    # fds

    # """TEST GRAPH HERE"""
    # # We want
    # dico = {}
    # dico_aug = {}
    # with torch.no_grad():
    #     for i, sentence in enumerate(dfVal['sentence']):
    #         # if i==10:
    #         #     break
    #         exp = explainer_baseline.explain_instance(sentence, predict_baseline, num_features=3)
    #         exp_aug = explainer_augmented.explain_instance(sentence, predict_aug, num_features=3)
    #         # probs=c_baseline.predict_proba([sentence])[0]#[0, 1])
    #         for elem in exp.as_list():
    #             try:
    #                 dico[elem[0]].append(elem[1])
    #             except:
    #                 dico[elem[0]] = [elem[1]]
    #         for elem in exp_aug.as_list():
    #             try:
    #                 dico_aug[elem[0]].append(elem[1])
    #             except:
    #                 dico_aug[elem[0]] = [elem[1]]
    #         # explainer_augmented.explain_instance(sentence, c_augmented.predict_proba, num_features=3)
    # # print(dico)
    # # print(dico_aug)
    # dico = {key: value for key, value in dico.items() if len(value) > 6}
    # print(dico.keys())
    # dico_aug = {key: value for key, value in dico_aug.items() if len(value) > 6}
    # print(len(dico))
    # print(len(dico_aug))
    # key = list(dico.keys())
    # values = list(dico.values())
    # values = [np.mean(val) for val in values]
    #
    # for word_aug in dico_aug.keys():
    #     if word_aug not in key:
    #         key.append(word_aug)
    #         dico[word_aug] = [0]
    #
    # for word in key:
    #     if word not in list(dico_aug.keys()):
    #         dico_aug[word] = [0]
    #
    # y_baseline = [np.mean(dico[k]) for k in key]
    # y_aug = [np.mean(dico_aug[k]) for k in key]
    #
    # print(y_baseline)
    # print(values)
    # data = [23, 24, 5]
    #
    # Fig = plt.figure()
    # plt.xticks(range(len(key)), key, rotation='vertical')
    # plt.ylabel("Correlation with the positive class")
    # plt.bar(np.arange(len(key)) - 0.2, y_baseline, 0.4, color='red', label='baseline')
    # plt.bar(np.arange(len(key)) + 0.2, y_aug, 0.4, color='blue', label='augmented')
    # plt.legend()
    # plt.savefig(f'Graphes/LIME_{argdict["dataset"]}_{argdict["dataset_size"]}_{argdict["split"]}_bert.png',
    #             bbox_inches='tight')
    # fdas
    print("------------------")
    print("Selecting 5 examples that the labels was wrong and became right")
    examples = np.where(bad_to_correct == 1)[0][:5]
    # print(examples)
    for idx in examples:
        # print(idx)
        print("BASELINE")
        print(dfVal['sentence'][idx])
        exp = explainer_baseline.explain_instance(dfVal['sentence'][idx], predict_baseline, num_features=6)
        print('Document id: %d' % idx)
        pred_baseline = predict_baseline([dfVal['sentence'][idx]]).squeeze(0)
        print(f'Probability({labels[torch.argmax(pred_baseline)]}) =', torch.max(pred_baseline))
        print('True class: %s' % labels[dfVal['label'][idx]])
        print(exp.as_list())
        print("\nAUGMENTED")
        print(dfVal['sentence'][idx])
        exp = explainer_augmented.explain_instance(dfVal['sentence'][idx], predict_aug,
                                                   num_features=6)
        print('Document id: %d' % idx)
        pred_aug = predict_aug([dfVal['sentence'][idx]]).squeeze(0)
        print(f'Probability({labels[torch.argmax(pred_aug)]}) =', torch.max(pred_aug))
        print('True class: %s' % labels[dfVal['label'][idx]])
        print(exp.as_list())
        print('*****************')

    print("Selecting 5 examples that the labels was correct and became wrong")
    examples = np.where(correct_to_bad == 1)[0][:5]
    for idx in examples:
        print("BASELINE")
        print(dfVal['sentence'][idx])
        exp = explainer_baseline.explain_instance(dfVal['sentence'][idx], predict_baseline, num_features=6)
        print('Document id: %d' % idx)
        pred_baseline = predict_baseline([dfVal['sentence'][idx]]).squeeze(0)
        print(f'Probability({labels[torch.argmax(pred_baseline)]}) =', torch.max(pred_baseline))
        print('True class: %s' % labels[dfVal['label'][idx]])
        print(exp.as_list())
        print("\nAUGMENTED")
        print(dfVal['sentence'][idx])
        exp = explainer_augmented.explain_instance(dfVal['sentence'][idx], predict_aug,
                                                   num_features=6)
        print('Document id: %d' % idx)
        pred_aug = predict_aug([dfVal['sentence'][idx]]).squeeze(0)
        print(f'Probability({labels[torch.argmax(pred_aug)]}) =', torch.max(pred_aug))
        print('True class: %s' % labels[dfVal['label'][idx]])
        print(exp.as_list())
        print('*****************')

def lime_dan(argdict):
    """Lime analysis with xgboost"""
    labels = argdict['categories']
    # labels = [f"{label}_selected" for label in labels] + [f"{label}_generated" for label in labels]
    SelectedData = pd.read_csv( f"{argdict['pathDataAdd']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}/train.tsv", sep='\t')
    dfVal = pd.read_csv(f'{argdict["pathDataOG"]}/{argdict["dataset"]}/dev.tsv', sep='\t')
    true_labels = dfVal['label']
    true_labels_train = SelectedData['label']
    print(f"Accuracy on the train set for the baseline: {accuracy_score(true_labels_train, predictions_train)}")
    print(f"Accuracy on the validation set for the baseline: {accuracy_score(true_labels, predictions)}")