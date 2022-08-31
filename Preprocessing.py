import pandas as pd

def preprocess_trec():
    """Preprocess TREC in same format as other dataset, by putting dataframe in data/TREC6
    Following the other datasets the dataframe will be composed of sentence, label"""
    train=open("/media/frederic/DAGlue/data/TREC6/train_5500.label", "r", errors="replace")
    dic={'sentence':{}, 'label':{} }
    labels={"ABBR":0, "DESC":1, "ENTY":2, "HUM":3, "LOC":4, "NUM":5}

    for i, ll in enumerate(train.readlines()):
        label=ll.split()[0].split(':')[0]
        sentence=" ".join(ll.split()[1:])
        dic['label'][i]=labels[label]
        dic['sentence'][i]=sentence
    df=pd.DataFrame.from_dict(dic)
    df.to_csv('/media/frederic/DAGlue/data/TREC6/train.tsv', sep='\t', index=False)

    dev = open("/media/frederic/DAGlue/data/TREC6/TREC_10.label", "r", errors="replace")
    dic = {'sentence': {}, 'label': {}}
    for i, ll in enumerate(dev.readlines()):
        label = ll.split()[0].split(':')[0]
        sentence = " ".join(ll.split()[1:])
        dic['label'][i] = labels[label]
        dic['sentence'][i] = sentence
    df = pd.DataFrame.from_dict(dic)
    df.to_csv('/media/frederic/DAGlue/data/TREC6/dev.tsv', sep='\t', index=False)


preprocess_trec()