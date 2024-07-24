# from pathlib import Path

import pandas as pd

import spacy
from spacy.cli.train import train
from spacy.cli.evaluate import evaluate
from spacy.cli.project.run import project_run
from spacy.cli.project.assets import project_assets
from spacy.util import compounding, minibatch
from spacy.training import Example

from preprocess import preprocessing

def convert(nlp, data, outfile):
    db = spacy.tokens.DocBin()
    for doc, label in nlp.pipe(data, as_tuples=True):
        doc.cats["POS"] = label == 1
        doc.cats["NEG"] = label == 0
        db.add(doc)
    
    db.to_disk(outfile)
    # print(f" Processed {len(db)} documents: {outfile}")
    # return db


def spacy_prepare_training(df, DATA_DIR, params):
    print(f'\nPreprocessing dataset...')
    df["text_clean"] = df["text"].apply(preprocessing)
    print(f' Preprocessed!') # \n{df.head()}')

    print(f'\nBalancing dataset...')

    # as sentiments 0 & 1 are unbalanced (~ 1 : 3-5 ), let's mix a good proportion manually
    dataset0 = list(df[df["sentiment"]==0][["text_clean", "sentiment"]].sample(frac=1).itertuples(index=False, name=None))
    dataset1 = list(df[df["sentiment"]==1][["text_clean", "sentiment"]].sample(frac=1).itertuples(index=False, name=None))

    min_len = min(len(dataset0), len(dataset1))
    print(' by', df['sentiment'].value_counts().to_string())

    train_data = dataset0[:int(min_len*0.8)] + dataset1[:int(min_len*0.8)]
    dev_data = dataset0[int(min_len*0.8):int(min_len*1.00)+1] + dataset1[int(min_len*0.8):int(min_len*1.00)+1]

    print(f"\nTotal: {df.shape[0]} // Min subset: {min_len} // Split: train {len(train_data)} + dev {len(dev_data)}") #  // Test: {len(test_data)}

    params['min_subset'] = min_len
    nlp = spacy.blank('en')
    
    convert(nlp, train_data, f"{DATA_DIR}/corpus/train.spacy")
    convert(nlp, dev_data, f"{DATA_DIR}/corpus/dev.spacy")
    print(f' Data saved to {DATA_DIR}/corpus/ and ready for training.\n')

    return params


def spacy_run_project(DATA_DIR, params):
    #### Option 1 - run pipeline via project.yml file in DATA_DIR
    # root = Path(__file__).parent
    # root = Path(DATA_DIR)
    # project_assets(root)
    # project_run(root, "all", capture=True)

    #### Option 2 - run each step via spacy.cli calls + config
    # with ability to **override parameters** -> HPO
    # + see training progress log

    overrides={'paths.train': f'{DATA_DIR}/corpus/train.spacy', 
            'paths.dev': f'{DATA_DIR}/corpus/dev.spacy',
            }

    hyperparam_list = ['training.optimizer.learn_rate', 'training.optimizer.L2', 'training.dropout', 
                        'training.optimizer.use_averages', 'training.optimizer.eps', ]

    # overriding HP present in params
    for hyperparam in hyperparam_list:
        if hyperparam in params:
            overrides[hyperparam] = params[hyperparam]
    
    train(f'{DATA_DIR}/configs/config.cfg', 
            output_path=f'{DATA_DIR}/training/',
            overrides=overrides)
    
    metrics = evaluate(model=f'{DATA_DIR}/training/model-best/',
            data_path=f'{DATA_DIR}/corpus/dev.spacy',
            output=f'{DATA_DIR}/training/metrics.json'
            )
    # metrics = spacy_read_metrics(DATA_DIR)
    print('================\nCategorization scores:', metrics['cats_score'])
    print(' auc_per_type:', metrics['cats_auc_per_type'])

    return metrics


if __name__ == '__main__':
    years = [2024]
    df = pd.DataFrame({'text':[
                                'absolutely aweful book',
                                'book is bad - too short, too boring, waste of money',
                                'story is boring, I didn\'t like it',
                                'good book, helpful for beginners',
                                'absolutely amazing system, author is my new hero, strongly recommend',
                                'horrible, don\'t even think to buy it',
                                'total frustration, I didn\'t finish it',
                                'beginning was promising, but author failed to develop it, hard to read',
                                'quite good book, was useful for me',
                                'loved the story and characters, respect to the author, best book of the month',
                                ], 
                    'rating':[1, 2, 3, 4, 5,1, 2, 3, 4, 5,],
                    'sentiment':[0, 0, 0, 1, 1,0, 0, 0, 1, 1,],
                    })
    DATA_DIR = './spacy_proj'
    params = {'optimizer':'Adam.v1', 
                'training.dropout': 0.1,
                'training.optimizer.eps': 0.00000001,
                'training.optimizer.learn_rate': 0.0005,
                'training.optimizer.use_averages': True,
                'years': str(years),
                }
    params = spacy_prepare_training(df, DATA_DIR, params)
    metrics = spacy_run_project(DATA_DIR, params)
