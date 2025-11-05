from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import BitsAndBytesConfig
import torch


embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5', device = 'cpu')
Settings.embed_model = embed_model

bnb_config = BitsAndBytesConfig(
  load_in_8bit=True,
  bnb_8bit_compute_dtype=torch.float16,
  bnb_8bit_use_double_quant=True,
  bnb_8bit_quant_type="nf4",
)  

llm = HuggingFaceLLM(
  model_name = 'HuggingFaceH4/zephyr-7b-alpha',
  tokenizer_name = 'HuggingFaceH4/zephyr-7b-alpha',
  query_wrapper = '{query_str}',
  context_window = 3900,
  max_new_tokens = 256, 
  model_kwargs = {'quantization_conifg' : bnb_config},
  generate_kwargs = {'use_cache' : True, 'temeprature' : 0.8, 'top_k' : 50, 'top_p' : 0.95 }
  device_map = 'auto'
)
Settings.llm = llm


docx = SimpleDirectoryReader(
    './data',
    recursive = True
).load_data()
index = VectorStoreIndex.from_documents(docx)

chat_engine = index.as_chat_engine()
response = chat_engine.chat("who is Aryan Chauhan?")
print(response)
