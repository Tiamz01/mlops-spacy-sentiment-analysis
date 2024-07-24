import warnings
warnings.simplefilter("ignore", category=UserWarning)

import json
from time import time

import pandas as pd

import spacy

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import mlflow.spacy
EXPERIMENT_NAME = "Training Spacy model for sentiment analysis"
HPO_EXPERIMENT_NAME = "Testing trained Spacy models"

from prefect import flow, task

DEBUG = False

from preprocess import preprocessing
from predict import SENTIMENT_THRESHOLD, SENTIMENT_THRESHOLD2
from predict import spacy_get_sentiment, spacy_get_sentiment_preprocess, spacy_test_text, spacy_test_list
from train_model import spacy_prepare_training, spacy_run_project

nlp = spacy.blank("en")


def load_data_from_parquet(years=[]):
    print('\nLoading dataset...')

    DATA_DIR = './data/'
    # MASK = 'reviews-sentiment[*].parquet'
    try:
        df = pd.concat([pd.read_parquet(f'{DATA_DIR}reviews-sentiment[{year}].parquet') for year in years])
    except:
        print(f' local files not found, loading from repository...')
        df = pd.concat([pd.read_parquet(f'https://github.com/dmytrovoytko/reviews-sentiment-dataset/raw/master/reviews-sentiment%5B{year}%5D.parquet') for year in years])

    total_rows_number = df.shape[0]

    if years:
        df = df[df['review_year'].isin(years)]

    df['sentiment'] = 0 # set all negative
    df.loc[df.rating>=4, "sentiment"] = 1 # rating 4, 5 - positive

    # filtering out outliers by text length
    df['text_len'] = df["text"].apply(lambda x: len(x)//25*25) # 
    threshold1 = 25
    threshold2 = 3_000 # ~98%
    df = df[(df.text_len>=threshold1) & (df.text_len<=threshold2)  
                # & ~(df_y.rating==4)] # 4 are ~ tricky reviews - should be positive, 
                                     # but many people regret that it wasn't 'perfect' for 5
    ]

    if DEBUG:
        print('\n by', df['text_len'].value_counts().head(20).to_string())
        # Last in TOP 20 ~ text_len 500

    # truncate long reviews, without cutting last word
    TRUNCATE_LEN = 1000

    if DEBUG:
        df1 = df[(df["text"].str.len()>TRUNCATE_LEN)]
        print('before TRUNCATE', df1.shape[0]) #, df1["text"].head(2).to_list())  

    # take first 500 + last 500, without cutting words - usually (emotional) opinions begin/end reviews
    df['text'] = df['text'].apply(lambda x: ' '.join(x[:(TRUNCATE_LEN//2)].split(' ')[:-1]) +
                                                ' ' + ' '.join(x[-(TRUNCATE_LEN//2):].split(' ')[1:]) 
                                                if len(x) > TRUNCATE_LEN else x)

    if DEBUG:
        # checking no such long rows anymore
        df1 = df[(df["text"].str.len()>TRUNCATE_LEN)]
        print('after TRUNCATE', df1.shape[0], df1["text"].head(10).to_list())  

        df['text_len'] = df["text"].apply(lambda x: len(x)//25*25) #.str.len()
        print('\n by', df['text_len'].value_counts().head(50).to_string())


    print(f'\nFiltered number of records ({years}):', df.shape[0],'/',total_rows_number,
            '=', f'{df.shape[0]/total_rows_number*100:05.2f}%')
    print('\n by', df['rating'].value_counts().to_string())
    print('\n by', df['sentiment'].value_counts().to_string())
    # exit()
    return df

def spacy_read_metrics(DATA_DIR):
    with open(f'{DATA_DIR}/training/metrics.json') as f:
        metrics = json.load(f)
        return metrics


@task
def spacy_test_model(df, DATA_DIR, preprocess=False):
    t_start = time()
    nlp = spacy.load(f"{DATA_DIR}/training/model-best")
    print(f'\nTesting model {DATA_DIR}, preprocess: {preprocess}')


    # apply get_sentiment function
    if preprocess:
        df['spacy_sentiment'] = df.apply(lambda row: spacy_get_sentiment_preprocess(row['text'], nlp, False), axis=1)
    else:
        df['spacy_sentiment'] = df.apply(lambda row: spacy_get_sentiment(row['text'], nlp, False), axis=1)
    print(f'\nApply spacy_Sentiment finished in {(time() - t_start):.3f} second(s)')
    # print(df.head())

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    conf_matrix = confusion_matrix(df['sentiment'], df['spacy_sentiment'])
    print(conf_matrix)
    tn, fp, fn, tp = conf_matrix.ravel()
    print('tn, fp, fn, tp =', tn, fp, fn, tp)

    print(classification_report(df['sentiment'], df['spacy_sentiment']))

#########

@task
def spacy_run_experiment(df, DATA_DIR, params, run_name):
    # mlflow.autolog() # as run is via cli call, nothing is detected -> log explicitly 
    t_start = time()
    print(f'\nStarting experiment [{run_name}] in {DATA_DIR}...')
    with mlflow.start_run(run_name=run_name) as run:
        params = spacy_prepare_training(df, DATA_DIR, params)
        mlflow.log_params(params)

        # train and save model (CLI)
        # python -m spacy init config  --lang en --pipeline textcat --optimize efficiency --force config.cfg
        # python -m spacy train config.cfg --paths.train ./train.spacy --paths.dev ./dev.spacy --output model --verbose
        metrics = spacy_run_project(DATA_DIR, params)

        # Log the spaCy model using mlflow
        nlp = spacy.load(f"{DATA_DIR}/training/model-best")
        mlflow.spacy.log_model(spacy_model=nlp, artifact_path="model") # /textcat
        mlflow.log_artifact(f'{DATA_DIR}/training/metrics.json')
        # metrics = spacy_read_metrics(DATA_DIR)
        # print(metrics)
        mlflow.log_metric('cats_score', metrics["cats_score"])
        mlflow.log_metric('cats_macro_f', metrics["cats_macro_f"])
        mlflow.log_metric('cats_macro_p', metrics["cats_macro_p"])
        mlflow.log_metric('auc_per_type_POS', metrics['cats_auc_per_type']['POS'])
        mlflow.log_metric('auc_per_type_NEG', metrics['cats_auc_per_type']['NEG'])
        mlflow.log_metric('speed', metrics['speed'])
        print('model_uri:', f"runs:/{run.info.run_id}/model")
        # print(f"Model saved in run {mlflow.active_run().info.run_uuid}\n")

    # mlflow.end_run()
    print(f'Experiment finished in {(time() - t_start):.3f} second(s)\n')

@task
def run_register_model(data_path: str, top_n: int =1):
    print('\nRegistering best model...')
    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.cats_score DESC"]
    )
    for run in runs:
        print(f'Best performing models from run_id {run.info.run_id}, {run.data.params}')
        # test model again?

    # Select the model with the highest test cats_score
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1, # 3 to see and compare
        order_by=["metrics.cats_score DESC"]
    )[0]
    print(f'\nBest results: run_id {best_run.info.run_id}, metrics {best_run.data.metrics}\nparams: {best_run.data.params}')

    # Register the best model TODO get last model version? compare, register only if better
    result = mlflow.register_model(
        f'runs:/{best_run.info.run_id}/model', "spacy-textcat-reviews"
    )
    print(f'\nModel registered: {result}')
    print(f' Name: {result.name} Version: {result.version} run_id: {result.run_id}')

    # model_uri = f"models:/{model_name}/{model_version}"
    # model = mlflow.sklearn.load_model(model_uri)
    nlp = mlflow.spacy.load_model(f'runs:/{best_run.info.run_id}/model') # -best
    if DEBUG:
        print(nlp.config)

    # copy best model to ./model dir
    import shutil
    src = result.source.replace('file:/', '')
    dest = './model/model-best'
    shutil.copytree(src, dest, dirs_exist_ok=True)  # 3.8+ only!
    print(f'\nModel saved to {dest}')
    
    EXTRA_TEST = True
    if EXTRA_TEST:
        print(f'\n\nExtra tests!')
        df = load_data_from_parquet([2022]) # train on 2020/2021, test on 2022
        DATA_DIR = './spacy_proj'
        spacy_test_model(df, DATA_DIR, preprocess=False)

        df = load_data_from_parquet([2021]) # train on 2020/2021, test on 2022
        DATA_DIR = './spacy_proj'
        spacy_test_model(df, DATA_DIR, preprocess=False)


@flow
def ml_workflow():
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    years=[2021] # train on 2020/2021, test on 2022
    df = load_data_from_parquet(years)
    # defaults:
        # 'training.dropout': 0.1,
        # 'training.optimizer.learn_rate': 0.001,
        # 'training.optimizer.use_averages': False,
        # 'training.optimizer.eps': 0.00000001,

    # Training with different hyper parameters
    for L2 in [0.0001]:
        for eps in [0.00000001, ]:
            for learn_rate in [ 0.00005, 0.0001, 0.0004,]:
                DATA_DIR = './spacy_proj'
                params = {'optimizer':'Adam.v1', 
                            'training.dropout': 0.1, # 0.05 0.1 0.2
                            'training.optimizer.L2':  L2, # 0.00001,   , 0.001
                            'training.optimizer.eps': eps, # 0.000000005, 0.00000001, 0.00000002
                            'training.optimizer.learn_rate': learn_rate, # 0.00001, 0.00002, 0.00003, 0.00005, 0.0001,   , 0.0002
                            'training.optimizer.use_averages': False,
                            'years': str(years),
                            }
                spacy_run_experiment(df, DATA_DIR, params=params, 
                                    run_name=f"optimizer {params['optimizer']}, data {params['years']}")

                # different optimizer in a separate dir
                DATA_DIR = './spacy_proj2'
                params = {'optimizer':'RAdam.v1', 
                            'training.dropout': 0.1, # 0.05 0.1 0.2
                            'training.optimizer.L2':  L2, # 0.00001,   , 0.001
                            'training.optimizer.eps': eps, # 0.000000005, 0.00000001, 0.00000002
                            'training.optimizer.learn_rate': learn_rate, # 0.00005, 0.0001,   , 0.0002
                            'training.optimizer.use_averages': False,
                            'years': str(years),
                            }
                spacy_run_experiment(df, DATA_DIR, params=params, 
                                    run_name=f"optimizer {params['optimizer']}, data {params['years']}")



    # Testing model on different data
    df = load_data_from_parquet([2022]) # train on 2020/2021, test on 2022

    DATA_DIR = './spacy_proj'
    spacy_test_model(df, DATA_DIR, preprocess=True)
    if DEBUG:
        spacy_test_model(df, DATA_DIR, preprocess=False)

    # register_model
    DATA_DIR = './spacy_proj'
    run_register_model(DATA_DIR, 3)


if __name__ == '__main__':

    # run Prefect workflow
    ml_workflow()


    # Extra Test model
    TESTING_MODE1 = True # False
    if TESTING_MODE1:
        df = load_data_from_parquet([2022]) # train on 2020/2021, test on 2022

        DATA_DIR = './spacy_proj'
        spacy_test_model(df, DATA_DIR, preprocess=True)
        spacy_test_model(df, DATA_DIR, preprocess=False)

        DATA_DIR = './spacy_proj2'
        spacy_test_model(df, DATA_DIR, preprocess=True)
        spacy_test_model(df, DATA_DIR, preprocess=False)

    TESTING_MODE2 = False # True
    if TESTING_MODE2:

        MODEL_DIR = './model/model-best/model.spacy/'

        nlp = spacy.load(f"{MODEL_DIR}")
        print(f'\nTesting model {MODEL_DIR}')
        text = 'I liked that book it was fun'
        spacy_test_text(nlp, text, verbose=True)

        texts = ['Text is too short. I didn\'t like it',
                "This story has given me incites into a very difficult situation that exist because politicians in both parties have made it a my way or the highway position on imigration. The number of illegals in the country is far larger than the 12 million number being used and the percentage of Latinos is about 25 to 30% of the total. We lack the resources to vet properly the illegals that are here now and we have to acknowledge the criminals that are here NOW and we don't get ANY substantial help from the countries south of the border due, in part, to corruption and influence by the cartels in those governments. In closing, I don't have a simple solution, as there is none, but we must stop reacting with JUST emotion, we must challenge our perceptions, on both sides, to wrestle toward a solution",
                "When I picked this book up, I really wanted to like it. Having thoroughly enjoyed Ender's Game, I was very excited to experience this story from another perspective.<br /><br />Unfortunately, I found this story a little disturbing as it manages to undermine Ender and his accomplishments almost entirely. The point continuously made through Ender's Game was that he would have to rely fully on his own ability and would get no help from any source. This book clearly shatters that notion and goes on to  show that Bean is the force behind much of Ender's success, and that Ender would not have attained all he had without Bean's exceptional intellect. One case in point: Bean was responsible for hand picking Ender's Dragon Army while it took time for Ender to realize their full worth as soldiers.<br /><br />The notion of Bean's enhanced intellectual ability reads like a cheap literary trick to justify the level of knowledge and insight the character would continuously have throughout the book. Bean is portrayed as being deficient in only one area- the ability to create almost fanatical loyalty from his subordinates. This deficiency is explained in his relationship with Achilles and inability to take a leadership role within the ""family."" This; however, is a very poor comparison to Battle School, where the interactions between students, while sometimes contentious, are incorrectly framed within Bean's past experiences of life and death situations- which he obviously knows.<br /><br />In fact, in Bean's first interaction with Ender, he shows an uncanny ability to analyze and understand Ender's leadership deficiencies and their consequences, even while Ender struggled with them. This indicates that Bean has far greater understanding of leadership than Ender, and we already know he has the necessary judgment to employ this understanding. In fact, by this point, Bean has already shown himself to be a leader in fact, if not in title or authority. His rant about leading a Toon was childishly out of character for Bean and included solely for the sake of being in sync with the interaction in Ender's Game. To further reinforce the issue of Bean's leadership deficiencies, it is continuously pointed out that his physical stature is an issue. This is a moot point as Ender encountered the same issue, and while Ender handled the Bonzo interaction with physical violence, Bean was no less successful with Achilles and was able to efficiently and effectively dispatch his protagonist. (It can be argued that Achilles was a dumbed down copy of Peter Wiggins, violent,ambitious, patient, intelligent, vengeful, etc etc etc.)<br /><br />In the end, I think Bean is portrayed as being better than Ender in every way with the exception that he was the intellectual superior of the teachers as well, and therefore, not to be trusted. This is a pretty weak argument at best, as it reduces the teachers to being petty and selecting Ender only because they liked him best. This thoroughly destroys the image that they were reluctant to damaging a child but did so in the name of humanity on the one hand, while on the other, excluding a better candidate because of personal preference",
                'I didn\'t like reading the book',
                'don\'t buy it',
                'I do not dislike cabin cruisers',
                'Disliking watercraft is not really my thing',
                'Sometimes I really hate RIBs.',
                'I\'d really truly love going out in this weather! ',
                'The movie is surprising, with plenty of unsettling plot twists',
                'You should see their decadent dessert menu.',
                'I love my mobile but would not recommend it to any of my colleagues',
                'I dislike old cabin cruisers',
                ]
        spacy_test_list(nlp, texts, verbose=True)  

    REGISTER_MODEL = False # False # True
    if REGISTER_MODEL:
        DATA_DIR = './spacy_proj'
        run_register_model(DATA_DIR)


