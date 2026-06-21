from classes import Embedding, PersistenceDiagram
from dotenv import load_dotenv
import argparse, os

load_dotenv()
data_dir = os.getenv('DATA_DIR')
save_dir = os.getenv('SAVE_DIR')
parser = argparse.ArgumentParser()

parser.add_argument('model_name')
parser.add_argument('file_path')
parser.add_argument('group_name')
parser.add_argument('dataset_name')

args = parser.parse_args()
model_name = args.model_ame
file_path = args.file_path
group_name = args.group_name
dataset_name = args.dataset_name

embedding = Embedding(model_name=model_name)
embedding.embed(file_path=f'{data_dir}/{file_path}')
embedding.save(file_path=f'{data_dir}/{file_path}')

pd = PersistenceDiagram(data=embedding.embeddings)
pd.wasserstein(save=True)

