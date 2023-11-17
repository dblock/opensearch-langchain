#!/usr/bin/env python3

from os import environ
from typing import List
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.schema.embeddings import Embeddings

fake_texts = ["foo", "bar", "baz"]

class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(1.0)] * 9 + [float(i)] for i in range(len(texts))]

    def embed_query(self, text: str) -> List[float]:
        return [float(1.0)] * 9 + [float(0.0)]
    
docsearch = OpenSearchVectorSearch.from_texts(
    fake_texts, 
    FakeEmbeddings(), 
    opensearch_url=ENV['ENDPOINT'], 
    verify_certs=False, 
    http_auth=("admin", "admin")
)

OpenSearchVectorSearch.add_texts(
    docsearch, fake_texts, vector_field="my_vector", text_field="custom_text"
)
