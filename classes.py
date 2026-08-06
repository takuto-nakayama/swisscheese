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
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except:
            pass


    def embed_dynamic(self, file_name:str, batch:int):
        start = datetime.now()
        print(f'start embedding: {file_name}')

        with open(f'{data_dir}/{file_name}.txt') as f:
            text = f.readlines()
            text = [t.strip() for t in text]
        text = sorted(text, key=len)

        self.embeddings = []

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
        time = datetime.now() - start
        print(f'embedding: {file_name} ({time.seconds} seconds)')


    def embed_dynamic_wiki(self, config:str, batch:int, num_samples:int, seed:int):
        start = datetime.now()
        print(f'start embedding (wiki): {config}')
        dataset = load_dataset(
            'wikimedia/wikipedia',
            config,
            split='train'
            )
        random.seed(seed)
        indices = sorted(random.sample(range(0,len(dataset)),k=num_samples))
        articles = dataset.select(indices)
        self.embeddings = []
        sentences = []
        for article in articles:
            for paragraph in article['text'].split('\n'):
                if paragraph.strip():
                    sentences.append(paragraph.strip())
        sentences = sorted(sentences, key=len)

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

        self.embeddings = np.vstack(self.embeddings)
        time = datetime.now() - start
        print(f'config:{config}, length:{len(self.embeddings)}, duration:{time.seconds} seconds')


    def embed_fasttext_model(self, id:str, num_samples:int, seed:int):
        self.model = fasttext.load_model(f'{model_dir}/cc.{id}.300.bin')
        random.seed(seed)
        input_matrix = self.model.get_input_matrix()
        indices = random.sample(range(len(input_matrix)), k=num_samples)
        self.embeddings = input_matrix[indices]



class PersistenceDiagram:
    def __init__(self, embeddings, seed:int, num_samples:int):
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
        start = datetime.now()
        print('start pedg.')
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
        time = datetime.now() - start
        print(f'pedg ({time.seconds} seconds)')


class Distance:
    def __init__(self, dir_name:str, save_name:str):
        self.dir_name = dir_name
        self.save_name = save_name
        self.list_pds = sorted(os.listdir(f'{pd_dir}/{self.dir_name}'))
        self.D_h0 = np.empty((len(self.list_pds), len(self.list_pds)))
        self.D_h1 = np.empty((len(self.list_pds), len(self.list_pds)))


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


    def get_sliced_wasserstein(self, p:int, eps:float, z:float, n_directions:int, seed:int, max_directions:int):
        start = datetime.now()
        results_h0 = np.empty((len(self.list_pds), len(self.list_pds), 4))
        results_h1 = np.empty((len(self.list_pds), len(self.list_pds), 4))
        dgms_all = {}
        for pedg in self.list_pds:
            with open(f'{pd_dir}/{self.dir_name}/{pedg}', 'rb') as f:
                dgms_all[pedg] = pickle.load(f)['dgms']
        dgms_all = {
            k:
            [
                v[0][np.isfinite(v[0]).all(axis=1)],
                v[1][np.isfinite(v[1]).all(axis=1)]
            ]
            for k, v in dgms_all.items()
        }

        for i, lang_i in enumerate(self.list_pds):
            start_ij = datetime.now()
            print(f'{lang_i} started')
            dgms_i = dgms_all[lang_i]
            h0_i = dgms_i[0]
            h1_i = dgms_i[1]

            for j in range(i + 1, len(self.list_pds)):
                dgms_j = dgms_all[self.list_pds[j]]
                h0_j = dgms_j[0]
                h1_j = dgms_j[1]
                
                pilot_dist_h0, _, _ = _swd(X=h0_i, Y=h0_j, p=p, n_directions=n_directions, seed=seed)
                numdir_h0 = _estimate_directions(pilot_dist_h0, p=p, eps=eps, z=z, max_directions=max_directions)
                _, swd_h0, se_h0 = _swd(h0_i, h0_j, n_directions=numdir_h0, p=p,  seed=seed)
                ci_low_h0, ci_high_h0 = swd_h0 - z * se_h0, swd_h0 + z * se_h0

                pilot_dist_h1, _, _ = _swd(h1_i, h1_j, p=p, n_directions=n_directions, seed=seed)
                numdir_h1 = _estimate_directions(pilot_dist_h1, p=p, eps=eps, z=z, max_directions=max_directions)
                _, swd_h1, se_h1 = _swd(h1_i, h1_j, n_directions=numdir_h1, p=p, seed=seed)
                ci_low_h1, ci_high_h1 = swd_h1 - z * se_h1, swd_h1 + z * se_h1

                self.D_h0[i,j] = self.D_h0[j,i] = swd_h0
                self.D_h1[i,j] = self.D_h1[j,i] = swd_h1
                results_h0[i,j] = results_h0[j,i] = [swd_h0, se_h0, ci_low_h0, ci_high_h0]
                results_h1[i,j] = results_h1[j,i] = [swd_h1, se_h1, ci_low_h1, ci_high_h1]

            time =  datetime.now() - start_ij
            print(f'{lang_i} is done. ({time.seconds} seconds)')

        df_h0 = pd.DataFrame(
            self.D_h0,
            columns=self.list_pds,
            index=self.list_pds
            )
        df_h1 = pd.DataFrame(
            self.D_h1,
            columns=self.list_pds,
            index=self.list_pds
            )
        df_h0.to_csv(f'{ws_dir}/{self.dir_name}/{self.save_name}-swd-h0.csv')
        df_h1.to_csv(f'{ws_dir}/{self.dir_name}/{self.save_name}-swd-h1.csv')
        with open(f'{ws_dir}/{self.dir_name}/{self.save_name}-results-h0.pkl', 'wb') as f:
            pickle.dump(results_h0, f)
        with open(f'{ws_dir}/{self.dir_name}/{self.save_name}-results-h1.pkl', 'wb') as g:
            pickle.dump(results_h1, g)

        duration = datetime.now() - start
        print(f'process duration: {duration.seconds} seconds')


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


@staticmethod
def _random_directions(n_directions:int, dim:int, seed:int | None=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_directions, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions


def _swd(
        X,
        Y,
        n_directions,
        p,
        seed=None
        ):
    dim = X.shape[1]
    directions = _random_directions(n_directions, dim, seed=seed)  # (L, dim)

    X_proj = X @ directions.T   # (n_x, L)  ← forループなしで全方向まとめて射影
    Y_proj = Y @ directions.T   # (n_y, L)

    X_sorted = np.sort(X_proj, axis=0)   # 列(方向)ごとに一括ソート
    Y_sorted = np.sort(Y_proj, axis=0)

    n, m = X_sorted.shape[0], Y_sorted.shape[0]
    if n != m:
        common = max(n, m)
        t_common = np.linspace(0, 1, common)   # 1回だけ計算
        t_x, t_y = np.linspace(0, 1, n), np.linspace(0, 1, m)
        X_sorted = np.column_stack([np.interp(t_common, t_x, X_sorted[:, k]) for k in range(n_directions)])
        Y_sorted = np.column_stack([np.interp(t_common, t_y, Y_sorted[:, k]) for k in range(n_directions)])

    diff = np.abs(X_sorted - Y_sorted) ** p
    distances = diff.mean(axis=0)              # 各方向のW_p^p、(L,)
    swd = float(distances.mean() ** (1.0 / p))
    se_mean = np.std(distances) / np.sqrt(n_directions)
    grad = (1.0 / p) * np.mean(distances) ** (1.0 / p - 1.0)
    se_swd = abs(grad) * se_mean

    return distances, round(swd, 4), round(se_swd, 4)


def _estimate_directions(distances_pilot:np.ndarray, p:int, eps:float, z:float, max_directions:int):
    mu = distances_pilot.mean()
    sigma = distances_pilot.std()
    cv = sigma / mu
    directions = (z / (p * eps)) ** 2 * cv ** 2
    return int(min(np.ceil(directions), max_directions))
