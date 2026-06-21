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

	
	def pers_hom(self, thresh:float, num_samples:int, save:bool, file_path:str):
		with h5py.File(self.file_path, 'r') as f:
			self.data = f[self.dataset]
			samples = random.sample(range(0,len(self.data)), num_samples)
			sampled_embedding = [self.data[i] for i in samples]
			sampled_embedding = np.vstack(sampled_embedding)

			self.filtration = ripser(sampled_embedding,
									 thresh=thresh)
			self.dgms = self.filtration['dgms']

			if save:
				record = {
					'dgms':		self.dgms,
					'n_points':	len(self.data)
				}

				with open(f'{result_dir}/{file_path}.pkl', 'a') as f:
					pickle.dump(record, f)



class Distance:
	def __init__(self, pds_path:list, file_path:str):
		self.list_pds = os.listdir(f'{pd_dir}/{pds_path}')
		self.file_path = file_path


	def wasserstein(self):
		d_h0 = []
		d_h1 = []

		for i in self.list_pds:
			d_i_h0 = []
			d_i_h1 = []
			with open(f'{pd_dir}/{i}', 'rb') as f:
				dgms_i = pickle.load(f)
			for j in self.list_pds:
				if i == j:
					d_i_h0.append(0)
					d_i_h1.append(0)
				else:
					with open(f'{pd_dir}/{j}', 'rb') as f:
						dgms_j = pickle.load(f)
					d_i_h0.append(wasserstein(dgms_i[0], dgms_j[0]))
					d_i_h1.append(wasserstein(dgms_i[1], dgms_j[1]))

			d_h0.append(d_i_h0)
			d_h1.append(d_i_h1)
		
		with open(f'{result_dir}/{self.file_path}-h0.csv', 'w') as f:
			writer = csv.writer(f)
			writer.writerows()

		with open(f'{result_dir}/{self.file_path}-h1.csv', 'w') as f:
			writer = csv.writer(f)
			writer.writerows()