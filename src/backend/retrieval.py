import os
import yaml
from utils import TextCleaner, DocumentProcessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor


class VectorDBBuilder:
    def __init__(self, config_file):
        self.config = self.load_config(config_file)
        self.root_dir = self.config['root_dir']
        self.databases_dir = self.config['databases_dir']
        self.grades = self.config['grades']
        self.embedding_model = self.config['embedding_model']
        self.query_config = self.config['query_config']

        # Define the embedding model
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.embedding_model['name'])
        Settings.llm = None  # we won't use LlamaIndex to set up LLM
        Settings.chunk_size = self.embedding_model['chunk_size']
        Settings.chunk_overlap = self.embedding_model['chunk_overlap']

        # Initialize the text cleaner and document processor
        self.text_cleaner = TextCleaner()
        self.document_processor = DocumentProcessor(self.text_cleaner)

    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def build_vector_db(self, grade):
        # Define the directory for the current grade
        grade_dir = os.path.join(self.root_dir, grade)

        # Check if VectorDB for this grade already exists
        vector_db_path = os.path.join(self.databases_dir, f'{grade}_vector_db')
        if os.path.exists(os.path.join(vector_db_path, "docstore.json")):
            print(f"VectorDB for {grade} already exists. Skipping...")
            return

        # Get the list of files in the directory
        documents = SimpleDirectoryReader(grade_dir).load_data()

        # Clean and process the documents
        cleaned_documents = self.document_processor.process_documents(documents)

        # Create the index with the cleaned documents
        index = VectorStoreIndex.from_documents(cleaned_documents)

        # Save the index to the persist directory
        index.storage_context.persist(vector_db_path)
        print(f"Index saved to {vector_db_path}")

        # # Set number of docs to retreive
        # top_k = self.query_config['top_k']

        # # Configure retriever
        # retriever = VectorIndexRetriever(
        #     index=index,
        #     similarity_top_k=top_k,
        # )

        # # Assemble query engine
        # query_engine = RetrieverQueryEngine(
        #     retriever=retriever,
        #     node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=self.query_config['similarity_cutoff'])],
        # )

        # # Test the query engine
        # query = "how the family was travelling to Aqaba?"
        # response = query_engine.query(query)

        # # Reformat response
        # context = "Context:\n"
        # for i in range(top_k):
        #     try:
        #         context = context + response.source_nodes[i].text + "\n\n"
        #     except:
        #         continue

        # print(f'Grade: {grade}')
        # print(context)

    def build_all_vector_dbs(self):
        for grade in self.grades:
            self.build_vector_db(grade)

