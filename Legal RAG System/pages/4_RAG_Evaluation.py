# pages/4_RAG_Evaluation.py

import streamlit as st
import pandas as pd
import json
import os
from datasets import Dataset
from dotenv import load_dotenv

import nest_asyncio         
nest_asyncio.apply()

# RAGAS Imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# OpenRouter API configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="Evaluation - Legal RAG",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RAG Evaluation")
st.markdown("""
Load a session JSON file to view conversation turns and enter the *Ground Truth* (ideal answer) for complete evaluation.
""")

# =====================================================================
# 1. SETUP OPENROUTER (LLM) & HUGGINGFACE (EMBEDDINGS)
# =====================================================================

def get_evaluator_models():
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Use Groq for LLM 
    evaluator_llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key= GROQ_API_KEY,
        max_tokens=4096,
        max_retries=10
    )
    # Use HuggingFace embeddings (OpenRouter doesn't provide embeddings)
    # Note: Using show_progress=False and avoiding explicit device to prevent PyTorch issues
    try:
        evaluator_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={},  # Let sentence-transformers auto-detect device
            encode_kwargs={"normalize_embeddings": True}
        )
    except Exception as e:
        st.warning(f"Failed to load embeddings with auto device detection, trying CPU fallback: {e}")
        import torch
        # Force CPU with explicit tensor type
        evaluator_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": torch.device("cpu")},
            encode_kwargs={"normalize_embeddings": True}
        )
    
    return evaluator_llm, evaluator_embeddings

# =====================================================================
# 2. DATA LOADING AND EXTRACTION
# =====================================================================

def extract_turns_from_json(uploaded_file):
    """
    Load the JSON and return a flat list of dictionaries 
    containing question, answer, and contexts (for UI and GT editing).
    """
    try:
        json_data = json.load(uploaded_file)
    except json.JSONDecodeError:
        st.error("The uploaded file is not valid JSON.")
        return None
    
    flat_data = []
    sessions = json_data if isinstance(json_data, list) else [json_data]
        
    for session in sessions:
        history = session.get("history", [])
        
        for i in range(len(history) - 1):
            msg_user = history[i]
            msg_assistant = history[i+1]
            
            if msg_user["role"] == "user" and msg_assistant["role"] == "assistant":
                
                question = msg_user.get("content", "").strip()
                answer = msg_assistant.get("content", "").strip()
                contexts = msg_assistant.get("contexts", [])
                
                if question and answer:
                    flat_data.append({
                        "id": len(flat_data) + 1,
                        "question": question,
                        "answer": answer,
                        "contexts": contexts,
                        "initial_gt": msg_assistant.get("ground_truth", "")  # If already in JSON
                    })
    
    if not flat_data:
        st.error("No valid Question/Answer pair found in the JSON file.")
        return None
        
    # Initialize session state for editable Ground Truths
    if "evaluation_data" not in st.session_state or st.session_state.evaluation_data_id != uploaded_file.file_id:
        st.session_state.evaluation_data = flat_data
        st.session_state.evaluation_data_id = uploaded_file.file_id
    
    return st.session_state.evaluation_data

def prepare_ragas_dataset(evaluation_data_list):
    """
    Prepare the final Ragas Dataset using interface data.
    """
    ragas_data = []
    has_ground_truth = False
    
    for item in evaluation_data_list:
        # Clean contexts
        contexts = [c.strip() for c in item['contexts'] if c.strip()]
        if not contexts:
            contexts = ["No context retrieved"]
            
        gt = item.get("final_gt", "").strip()  # Use the final GT field
        
        if gt:
            has_ground_truth = True
            
        ragas_data.append({
            "question": item['question'],
            "answer": item['answer'],
            "contexts": contexts,
            "ground_truth": gt
        })
        
    df = pd.DataFrame(ragas_data)
    
    # Filter empty rows then create the Dataset
    df.dropna(subset=['question', 'answer'], inplace=True)
    if df.empty: return None, False
    
    return Dataset.from_pandas(df), has_ground_truth


# =====================================================================
# 3. USER INTERFACE AND EXECUTION LOGIC
# =====================================================================

# Sidebar for metrics info
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("The system uses GROQ (llama-3.1-8b-instant)* as judge.")
    st.markdown("---")
    st.markdown("##### Ragas Metrics:")
    st.markdown("Complete metrics are calculated only if *Ground Truth* is provided for at least one query.")

uploaded_file = st.file_uploader(
    "1. Upload session JSON file", 
    type=["json"]
)

if uploaded_file is not None:
    
    # PHASE 1: Loading and Display for GT input
    evaluation_data = extract_turns_from_json(uploaded_file)
    
    if evaluation_data:
        st.subheader("2. Ground Truth Input (Ideal Answer)")
        st.info("Fill in the 'Ground Truth' field with the perfect answer for each question. If left empty, Context Precision, Recall and Answer Correctness metrics will not be calculated for that query.")
        
        # Interface for entering GT
        for i, item in enumerate(evaluation_data):
            
            # Use an expander to compact the interface
            with st.expander(f"Query {item['id']}: {item['question'][:80]}..."):
                
                st.markdown(f"*User Question:* {item['question']}")
                st.markdown(f"*Chatbot Response:* {item['answer']}")
                
                # Widget for Ground Truth, linked to st.session_state
                # Use a unique key for the widget
                gt_key = f"gt_input_{item['id']}"
                
                # Initialize the value in session_state the first time
                if gt_key not in st.session_state:
                    st.session_state[gt_key] = item['initial_gt']
                    
                gt_value = st.text_area(
                    f"Ground Truth (Ideal Answer) for Query {item['id']}",
                    value=st.session_state[gt_key],
                    height=100,
                    key=gt_key
                )
                
                # Update session data with modified GT
                st.session_state.evaluation_data[i]['final_gt'] = gt_value
                
                # Display retrieved contexts (for Context Recall/Precision)
                with st.expander(f"📚 Retrieved Contexts ({len(item['contexts'])} documents)"):
                    for j, context in enumerate(item['contexts']):
                        st.text(f"Document {j+1}: {context[:300]}...")


        st.divider()
        
        # PHASE 2: Start Evaluation
        if st.button("🚀 3. Start Ragas Evaluation"):
            
            # Prepare the final Ragas dataset with entered GTs
            dataset_ragas, has_ground_truth = prepare_ragas_dataset(st.session_state.evaluation_data)
            
            if dataset_ragas is None:
                st.error("No valid data to evaluate after GT input.")
                st.stop()
            
            with st.spinner("Preparing OpenRouter models..."):
                llm, embeddings = get_evaluator_models()
                
            st.info(f"Evaluating {len(dataset_ragas)} interactions... *This may take some time*")
            
            progress_bar = st.progress(0)
            
            try:

                import asyncio
                import os
                if os.name == 'nt': # Se il sistema è Windows
                    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                # Definizione dinamica delle metriche (To have a complete evaluation use a powerful API like OpenAI)
                metrics_to_run = [faithfulness, answer_relevancy]
                if has_ground_truth:
                    metrics_to_run.extend([context_precision, context_recall, answer_correctness])
                
                results = evaluate(
                    dataset=dataset_ragas,
                    metrics=metrics_to_run,
                    llm=llm,
                    embeddings=embeddings,
                    raise_exceptions=True,
                    run_config=RunConfig(max_workers=1)
                )
                
                progress_bar.progress(100)
                
                # --- RESULTS DISPLAY ---
                st.divider()
                st.subheader("📈 Aggregated Evaluation Results")
                
                df_results = results.to_pandas()
                
                # Calculate Averages and prepare for Metric Cards
                metrics_display = {
                    'faithfulness': 'Faithfulness',
                    'answer_relevancy': 'Answer Relevancy',
                    'context_precision': 'Context Precision',
                    'context_recall': 'Context Recall',
                    'answer_correctness': 'Answer Correctness',
                }
                
                scores = {}
                for col_name, display_name in metrics_display.items():
                    if col_name in df_results.columns:
                        score = df_results[col_name].mean()
                        scores[display_name] = f"{score:.2f}"
                    else:
                        scores[display_name] = "N/A"

                # A. Metric Cards
                valid_scores = [(dn, s) for dn, s in scores.items() if s != "N/A"]
                cols = st.columns(max(len(valid_scores), 2))
                
                for i, (display_name, score) in enumerate(valid_scores):
                    cols[i].metric(display_name, score)
                
                # B. Detailed Table
                st.subheader("📝 Detailed Table per Interaction")
                
                # Columns to show in the table
                table_cols = ['question', 'answer', 'ground_truth', 'faithfulness', 'answer_relevancy', 
                              'context_precision', 'context_recall', 'answer_correctness', 'contexts']
                
                final_cols = [c for c in table_cols if c in df_results.columns]
                
                # Create a copy for display
                df_display = df_results[final_cols].copy()

                # 1. Start index from 1 instead of 0
                df_display.index = df_display.index + 1

                # 2. Name the index column "Id"
                df_display.index.name = "Query ID"
                
                # Display the table
                st.dataframe(
                    df_display,
                    width='stretch'
                )

                
            except Exception as e:
                st.error(f"Critical Ragas error during evaluation: {e}")
                import traceback
                traceback.print_exc()