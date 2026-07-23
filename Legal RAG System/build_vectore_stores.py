import sys
from pathlib import Path
from backend.document_loader import load_documents_with_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.config import RAGConfig
from backend.embeddings import get_embedding_model
from backend.vector_store import build_vector_store

ROOT = Path(__file__).parent  # la cartella dove sta build_vector_stores.py
DATA = ROOT / "Contest_Data"

divorce_codes_italy = load_documents_with_metadata(folders = [str(DATA/"Italy"/"Divorce_italy")],
                                                   country = "Italy", law = "Divorce", doc_type= "code")

divorce_codes_estonia = load_documents_with_metadata(folders = [str(DATA/"Estonia"/"Divorce_estonia")],
                                                   country = "Estonia", law = "Divorce", doc_type= "code")

divorce_codes_slovenia = load_documents_with_metadata(folders = [str(DATA/"slovenia"/"Divorce_slovenia")],
                                                   country = "Slovenia", law = "Divorce", doc_type= "code")

inheritance_codes_italy = load_documents_with_metadata(folders = [str(DATA/"Italy"/"Inheritance_italy")],
                                                   country = "Italy", law = "Inheritance", doc_type= "code")

inheritance_codes_estonia = load_documents_with_metadata(folders = [str(DATA/"Estonia"/"Inheritance_estonia")],
                                                   country = "Estonia", law = "Inheritance", doc_type= "code")

inheritance_codes_slovenia = load_documents_with_metadata(folders = [str(DATA/"slovenia"/"Inheritance_slovenia")],
                                                   country = "Slovenia", law = "Inheritance", doc_type= "code")

Italy_cases = load_documents_with_metadata(folders = [str(DATA/"Italy"/"Italian_cases_json_processed")],
                                        country= "Italy", doc_type="case")

Estonia_cases = load_documents_with_metadata(folders = [str(DATA/"Estonia"/"Estonian_cases_json_processed")],
                                        country= "Estonia", doc_type="case")

Slovenia_cases = load_documents_with_metadata(folders = [str(DATA/"slovenia"/"Slovenian_cases_json_processed")],
                                        country= "Slovenia", doc_type="case")


divorce_codes = divorce_codes_italy + divorce_codes_estonia + divorce_codes_slovenia

inheritance_codes = inheritance_codes_italy + inheritance_codes_estonia + inheritance_codes_slovenia

passed_cases = Italy_cases + Estonia_cases + Slovenia_cases

divorce_cases = [doc for doc in passed_cases if doc.metadata["law"] == "Divorce"]
inheritance_cases = [doc for doc in passed_cases if doc.metadata["law"] == "Inheritance"]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    length_function=len,
)

chunked_divorce_codes_documents = text_splitter.split_documents(divorce_codes)

chunked_inheritance_codes_documents = text_splitter.split_documents(inheritance_codes)

chunked_divorce_cases_documents = text_splitter.split_documents(divorce_cases)

chunked_inheritance_cases_documents = text_splitter.split_documents(inheritance_cases)

config = RAGConfig()
embedding_model = get_embedding_model(config)

build_vector_store(chunked_divorce_codes_documents, embedding_model, "vector_store/divorce_codes")
build_vector_store(chunked_inheritance_codes_documents, embedding_model, "vector_store/inheritance_codes")
build_vector_store(chunked_divorce_cases_documents, embedding_model, "vector_store/divorce_cases")
build_vector_store(chunked_inheritance_cases_documents, embedding_model, "vector_store/inheritance_cases")

