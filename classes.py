from datasets import load_dataset
from datetime import datetime
from dotenv import load_dotenv
from persim import plot_diagrams, wasserstein
from ripser import ripser
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv, fasttext, fasttext.util, h5py, os, pickle, random, re, time, torch



load_dotenv()
data_dir = os.getenv('DATA_DIR')
result_dir = os.getenv('RESULT_DIR')
emb_dir = os.getenv('EMB_DIR')
pd_dir = os.getenv('PD_DIR')
ws_dir = os.getenv('WS_DIR')
model_dir = os.getenv('MODEL_DIR')


class Embedding:
    def __init__(self, model_name:str=None):
        self.model_name = model_name


    def embed_fasttext(self, file_name:str, tokenizer_name:str):
        fasttext.util.download_model(self.lang, if_exists='ignore')
        self.model = fasttext.load_model(f'cc.{self.lang}.300.bin')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        with open(f'{file_name}.txt') as f:
            text = f.readlines()
        tokenized = [w for snt in text for w in self.tokenizer.tokenize(snt)]
        tokenized_sorted = sorted(set(tokenized))
        self.embeddings = []

        for t in tokenized_sorted:
            vec = self.model.get_word_vector(t)
            self.embeddings.append(vec)
        
        self.embeddings = np.vstack(self.embeddings)


    def embed_dynamic(self, file_name:str, batch:int):
        with open(f'{data_dir}/{file_name}.txt') as f:
            text = f.readlines()
            text = [t.strip() for t in text]
        text = sorted(text, key=len)

        self.embeddings = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        for i in range(0, len(text), batch):
            text_batched = text[i:min(i+batch, len(text))]
            inputs = self.tokenizer(
                text_batched,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512,
                return_special_tokens_mask=True
                )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            special_mask = inputs.pop('special_tokens_mask').bool()  
            with torch.no_grad():
                outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state
            attention_mask = inputs['attention_mask'].bool()
            keep = attention_mask & (~special_mask)
            self.embeddings.append(hidden[keep].cpu().numpy())

        self.embeddings = np.vstack(self.embeddings)


    def embed_fasttext_model(self, id:str, num_samples:int=10000, seed:int=42):
        self.model = fasttext.load_model(f'{model_dir}/cc.{id}.300.bin')
        random.seed(seed)
        input_matrix = self.model.get_input_matrix()
        indices = random.sample(range(len(input_matrix)), k=num_samples)
        self.embeddings = input_matrix[indices]


    def embed_dynamic_wiki(self, config:str, batch:int, num_samples:int=5000, seed:int=42):
        dataset = load_dataset(
            'wikimedia/wikipedia',
            config,
            split='train'
            )
        random.seed(seed)
        indices = sorted(random.sample(range(0,len(dataset)),k=num_samples))
        articles = dataset.select(indices)
        cnt = 0
        length = 0

        while length < num_samples:
            sentences = []
            paragraphs = articles[cnt]['text'].split('\n')
            for para in paragraphs:
                if para.strip():
                    sentences.append(para.strip())
            sentences = sorted(sentences, key=len)

            self.embeddings = []
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()

            for i in range(0, len(sentences), batch):
                snt_batched = sentences[i:min(i+batch, len(sentences))]
                inputs = self.tokenizer(
                    snt_batched,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_special_tokens_mask=True
                    )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                special_mask = inputs.pop('special_tokens_mask').bool()
                with torch.no_grad():
                    outputs = self.model(**inputs)
                hidden = outputs.last_hidden_state
                attention_mask = inputs['attention_mask'].bool()
                keep = attention_mask & (~special_mask)
                self.embeddings.append(hidden[keep].cpu().numpy())
                length += len(hidden[keep].cpu().numpy())

            cnt += 1

        self.embeddings = np.vstack(self.embeddings[:num_samples])
    


class PersistenceDiagram:
    def __init__(self, embeddings, seed:int=None, num_samples:int=10000):
        if seed is not None:
            random.seed(seed)
            indicess = sorted(random.sample(range(0,embeddings.shape[0]), k=num_samples))
            embeddings = embeddings[indicess]
        mean_norm = np.linalg.norm(embeddings, axis=1).mean()
        self.scaled_embeddings = embeddings / mean_norm
        dist_matrix = pairwise_distances(self.scaled_embeddings, metric='euclidean')
        mst = minimum_spanning_tree(csr_matrix(dist_matrix))
        self.max_mst = mst.data.max()


    def pers_homology(self, file_name:str=None):
        self.filtration = ripser(
            self.scaled_embeddings,
            thresh=self.max_mst,
            )
        self.dgms = self.filtration['dgms']
        record = {
            'dgms':self.dgms,
            'num_embeddings':len(self.scaled_embeddings)
        }

        with open(f'{pd_dir}/{file_name}.pkl', 'wb') as f:
            pickle.dump(record, f)



class Distance:
    def __init__(self, dir_name:str, save_name:str):
        self.dir_name = dir_name
        self.save_name = save_name
        self.list_pds = os.listdir(f'{pd_dir}/{self.dir_name}')
        self.D_h0 = np.zeros((len(self.list_pds), len(self.list_pds)))
        self.D_h1 = np.zeros((len(self.list_pds), len(self.list_pds)))


    def get_wasserstein(self):
        langs = [l.split('-')[-2]+l.split('-')[-1] for l in self.list_pds]
        dgms_all = {}
        for pedg, lang in zip(self.list_pds, langs):
            with open(f'{pd_dir}/{self.dir_name}/{pedg}', 'rb') as f:
                dgms_all[lang] = pickle.load(f)['dgms']

        def finite(dgm):
            return dgm[np.isfinite(dgm[:, 1])]
        dgms_all = {
            k: [finite(v[0]), finite(v[1])]
            for k, v in dgms_all.items()
        }

        for i, lang_i in enumerate(langs):
            dgms_i = dgms_all[lang_i]

            start_time = datetime.now().strftime('%Y%m%d%H%m%S')
            print(f'starts at {start_time[:4]}/{start_time[4:6]}/{start_time[6:8]}/{start_time[8:10]}:{start_time[10:12]}:{start_time[12:]}')
            start = time.time()

            for j in range(i + 1, len(langs)):
                dgms_j = dgms_all[langs[j]]
                d0 = wasserstein(dgms_i[0], dgms_j[0])
                d1 = wasserstein(dgms_i[1], dgms_j[1])
                self.D_h0[i, j] = self.D_h0[j, i] = d0
                self.D_h1[i, j] = self.D_h1[j, i] = d1
            elapsed = time.time()-start
            print(f'{langs[j].center(30)} is done. ({str(round(elapsed, 2)).center(10)}seconds.)')

        end_time = datetime.now().strftime('%Y%m%d%H%m%S')
        print(f'ends at {end_time[:4]}/{end_time[4:6]}/{end_time[6:8]}/{end_time[8:10]}:{end_time[10:12]}:{end_time[12:]}')

        df_h0 = pd.DataFrame(
            self.D_h0,
            columns=langs,
            index=langs
            )
        df_h1 = pd.DataFrame(
            self.D_h1,
            columns=langs,
            index=langs
            )
        df_h0.to_csv(f'{ws_dir}/{self.dir_name}/{self.save_name}-h0.csv')
        df_h1.to_csv(f'{ws_dir}/{self.dir_name}/{self.save_name}-h1.csv')


    def clustering(self):
        condensed_D_h0 = squareform(self.D_h0)
        condensed_D_h1 = squareform(self.D_h1)
        self.Z_h0 = linkage(condensed_D_h0, method='complete')
        self.Z_h1 = linkage(condensed_D_h1, method='complete')
        
        plt.figure(figsize=(6, 4))
        dendrogram(self.Z_h0, labels=self.list_pds)
        plt.ylabel('Distance')
        plt.savefig(f'{ws_dir}/{self.dir_name}/{self.save_name}-dendrogram-h0.png')

        plt.figure(figsize=(6, 4))
        dendrogram(self.Z_h1, labels=self.list_pds)
        plt.ylabel('Distance')
        plt.savefig(f'{ws_dir}/{self.dir_name}/{self.save_name}-dendrogram-h1.png')


    def msd_2d(self):
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        coords_h0 = mds.fit_transform(self.D_h0)
        coords_h1 = mds.fit_transform(self.D_h1)

        plt.figure(figsize=(6, 5))
        plt.scatter(
            coords_h0[:, 0],
            coords_h0[:, 1]
            )
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(f'{ws_dir}/{self.dir_name}/{self.save_name}-mds-h0.png')

        plt.figure(figsize=(6, 5))
        plt.scatter(
            coords_h1[:, 0],
            coords_h1[:, 1]
            )
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(f'{ws_dir}/{self.dir_name}/{self.save_name}-mds-h1.png')
