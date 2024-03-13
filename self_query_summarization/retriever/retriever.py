from typing import List

import random
import pandas as pd
import chromadb

from self_query_summarization.retriever.embeddings import GPTEmbeddings, SentenceTransformerEmbeddings


class ChromaRetriever:
        
    def __init__(self, 
               document_column: str,
               metadata_colums: List[str], 
               collection_name:str, 
               embedding_provider:str = 'sentence-transformers', 
               embedding_size:int = 1024, 
               similarity_metrics:str = "cosine"
    ):
        self.document_column = document_column
        self.collection_name = collection_name
        self.metadata_colums = metadata_colums
        self.embedding_provider = embedding_provider
        self.embedding_size = embedding_size
        self.similarity_metrics = similarity_metrics
        # initialize db client
        self.client = chromadb.Client()
        self._create_collection()
        self._select_embedding_model()
        
    def _validate_input(self, data):
        missing_columns = set(self.metadata_colums).difference(data.columns)
        if missing_columns:
            raise KeyError(f"meta data columns {missing_columns} not found in data columns")
        if self.document_column not in data.columns:
            raise KeyError(f"document column {self.document_column} not found in data columns")
                
    def _create_collection(self):
        # delete collection if it exists
        if self.collection_name in [c.name for c in self.client.list_collections()]:
            print(f"Deleting collection {self.collection_name}")
            self.drop_collection()
        print(f"Creating collection {self.collection_name}")
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.similarity_metrics} 
        )
        
    def _select_embedding_model(self):
        if self.embedding_provider == 'openai':
            self.embedding_model = GPTEmbeddings(self.embedding_size)
        elif self.embedding_provider == 'sentence-transformers':
            self.embedding_model = SentenceTransformerEmbeddings()
        else: 
            raise ValueError(f'Model {self.embedding_provider} is not implemented')
    
    def _parse_documents(self, data):
        return data[self.document_column].values.tolist()
    
    def _create_document_ids(self, data):
        return [f'{i}' for i in data['id']]
        
    def _parse_metadata(self, data):
        return data[self.metadata_colums].to_dict(orient="records")
        
    def generate_embeddings(self, documents: List[str]):
        print(f"Generating embeddings for {len(documents)} documents")
        return self.embedding_model.encode(documents)
        
    def upload(self, data: pd.DataFrame, n_samples:int = 1000):
        
        self._validate_input(data)
        
        if n_samples < len(data):
            subset_ids = random.sample(range(len(data)), n_samples)
            data = data.iloc[subset_ids]  
                 
        self.documents = self._parse_documents(data)
        self.ids = self._create_document_ids(data)
        self.metadatas = self._parse_metadata(data)
        self.embeddings = self.generate_embeddings(self.documents)
        
        self.collection.add(
            embeddings= self.embeddings,
            documents = self.documents,
            metadatas = self.metadatas,
            ids = self.ids
    )        
        
    def query(self, query_text:str, n_results:int=5, **kwargs):
        query_embeddings = self.embedding_model.encode(query_text)
        if len(query_embeddings)==1:
            # openai embeddings returns always a list of list also for a single string
            query_embeddings = query_embeddings[0]
        print("length of query_embeddings", len(query_embeddings))
        return self.collection.query(
            query_embeddings=query_embeddings, 
            n_results=n_results,
            include=["distances","metadatas","documents"],
            **kwargs
        )
        
    def drop_collection(self):
        self.client.delete_collection(self.collection_name)