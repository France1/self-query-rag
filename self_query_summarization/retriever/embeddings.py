from typing import List, Union
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer



class GPTEmbeddings:
    
    def __init__(self, size:int = 1024):
        self.size = size
        self._instantiate_model()
        
    def _instantiate_model(self):
        # TODO: try not using the langchain wrapper to see if it improves handling of single strings - see below
        self.model = OpenAIEmbeddings(
                model="text-embedding-3-large",
                dimensions=self.size
            )
        
    def encode(self, documents: Union[str,List[str]]):
        if isinstance(documents, str):
            # otherwise it return once embedding vector per string
            documents = [documents]
        return self.model.embed_documents(documents)
    

class SentenceTransformerEmbeddings:
    
    def __init__(self):
        self._instantiate_model()
        
    def _instantiate_model(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
    def encode(self, documents: Union[str,List[str]]):
        embeddings = self.model.encode(documents) 
        return embeddings.tolist()