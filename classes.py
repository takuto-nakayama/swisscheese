from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from ripser import ripser
from persim import plot_diagrams, wasserstein
import numpy as np
import csv, h5py, os, pickle, random, torch



load_dotenv()
data_dir = os.getenv('DATA_DIR')
result_dir = os.getenv('RESULT_DIR')
emb_dir = os.getenv('EMB_DIR')
pd_dir = os.getenv('PD_DIR')
ws_dir = os.getenv('WS_DIR')



class Embedding:
	def __init__(self, model_name:str):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = AutoModel.from_pretrained(model_name).to(self.device)
		self.model.eval()
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)


	def embed(self, file_path:str, batch:int):
		with open(f'{file_path}.txt') as f:
			text = f.readlines()
			text = [t.strip() for t in text]
		cycle = len(text) // batch
		additional = len(text) % batch
		start = 0
		self.embeddings = []

		for c in range(1, cycle+1):
			text_batched  =text[start:batch*c]
			inputs = self.tokenizer(text_batched,
									return_tensors='pt',
									truncation=True,
									padding=True,
									max_length=512)
			inputs = {k: v.to(self.device) for k, v in inputs.items()}
			
			with torch.no_grad():
				outputs = self.model(**inputs)
			hidden = outputs.last_hidden_state
			mask = inputs['attention_mask'].bool()
			self.embeddings.append(hidden[mask].cpu().numpy())
			start += batch

		self.embeddings = np.vstack(self.embeddings)


	def save(self, file_path:str, dataset:str):
		with h5py.File(f'{file_path}.h5', 'a') as f:
			f.create_dataset(dataset, data=self.embeddings)
		print(
			f'''{dataset} in {file_path}.h5
				{len(self.embeddings)} embeddings saved.
			''')



class PersistenceDiagram:
	def __init__(self, file_path:str, dataset:str):
		self.file_path = file_path + '.h5'
		self.dataset = dataset

	
	def pers_hom(self, thresh:float, metric:str, num_samples:int, save:bool, file_path:str):
		with h5py.File(self.file_path, 'r') as f:
			self.data = f[self.dataset]
			n = len(self.data)
			samples = sorted(random.sample(range(n), num_samples))
			sampled_embedding = self.data[samples]

			self.filtration = ripser(sampled_embedding,
									 thresh=thresh,
									 metric=metric)
			self.dgms = self.filtration['dgms']

			if save:
				record = {
					'dgms':			self.dgms,
					'n_points':		len(self.data),
					'num_samples':	num_samples
				}

				with open(f'{file_path}.pkl', 'wb') as f:
					pickle.dump(record, f)



class Distance:
    def __init__(self, pd_path: str, file_path: str):
        self.pd_path = pd_path
        self.list_pds = sorted(os.listdir(pd_path))
        self.file_path = file_path

    def get_wasserstein(self):
        names = self.list_pds
        n = len(names)
        dgms_all = {}
        for pd in names:
            with open(f'{self.pd_path}/{pd}', 'rb') as f:
                dgms_all[pd] = pickle.load(f)['dgms']

        def finite(dgm):
            return dgm[np.isfinite(dgm[:, 1])]
        def prune(dgm, eps=0.05):
            pers = dgm[:, 1] - dgm[:, 0]
            return dgm[pers > eps]

        dgms_all = {
            k: [prune(finite(v[0])), prune(finite(v[1]))]
            for k, v in dgms_all.items()
        }

        D_h0 = np.zeros((n, n))
        D_h1 = np.zeros((n, n))

        for i in range(n):
            dgms_i = dgms_all[names[i]]
            for j in range(i + 1, n):
                dgms_j = dgms_all[names[j]]
                d0 = wasserstein(dgms_i[0], dgms_j[0])
                d1 = wasserstein(dgms_i[1], dgms_j[1])
                D_h0[i, j] = D_h0[j, i] = d0
                D_h1[i, j] = D_h1[j, i] = d1
            print(f'{names[i]} is done.')

        self._save_csv(D_h0, f'{self.file_path}-h0.csv', names)
        self._save_csv(D_h1, f'{self.file_path}-h1.csv', names)

    def _save_csv(self, D, path, names):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([''] + list(names))
            for name, row in zip(names, D):
                writer.writerow([name] + list(row))