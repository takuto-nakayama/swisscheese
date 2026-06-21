from transformers import AutoTokenizer, AutoModel
from ripser import ripser
from persim import plot_diagrams
import numpy as np
import h5py, os, torch



load_dotenv()
data_dir = os.getenv('DATA_DIR')
save_dir = os.getenv('SAVE_DIR')



class Embedding:
	def __init__(self, model_name):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = AutoModel.from_pretrained(model_name).to(self.device)
		self.model.eval()
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)


	def embed(self, file_path):
		with open(f'{file_path}.txt') as f:
			text = f.readlines()
			text = [t.strip() for t in text]

		inputs = self.tokenizer(text,
								return_tensors='pt',
								truncation=True,
								padding=True,
								max_length=512)
		inputs = {k: v.to(self.device) for k, v in inputs.items()}
		
		with torch.no_grad():
			outputs = self.model(**inputs)
		hidden = outputs.last_hidden_state
		mask = inputs['attention_mask'].bool()
		self.embeddings = hidden[mask].cpu().numpy()


	def save(self, file_path, group, dataset):
		with h5py.File(f'{file_path}.h5', 'a') as f:
			f.create_dataset(f'{group}/{dataset}', data=self.embeddings)
		print(
			f'''{group}/{dataset} in {file_path}.h5
				{len(self.embeddings)} embeddings saved.
			''')



class PersistenceDiagram:
	def __init__(self, data):
		self.filtration = ripser(data)
		self.dgms = self.filtration['dgms']

	def save(self, file_path):
		

	def wasserstein(self):
		pass


	def bottleneck(self, save=False):
		pass
