import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import gradio as gr
from PIL import Image

# Configurazione Repository
BASE_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
LORA_MODEL_ID = "Eikichi22/vqa-flickr30k-llava" # Assicurati che questo sia il tuo repository HF pubblico!

# Su Hugging Face Spaces (free tier) avremo la CPU. Se c'è una GPU, la usa.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Caricamento modello su {DEVICE}...")

# 1. Caricamento del Processor
processor = LlavaProcessor.from_pretrained(BASE_MODEL_ID)

# 2. Configurazione Quantizzazione
# Nota: La quantizzazione 4-bit richiede GPU. Se lo spazio è CPU-only, il modello
# verrà caricato normalmente ignorando questa configurazione o usando la RAM.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16
)

# Caricamento Modello Base
model = LlavaForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config if DEVICE == "cuda" else None,
    device_map='auto' if DEVICE == "cuda" else None,
    low_cpu_mem_usage=True
)

# 3. Iniezione dell'Adapter LoRA
print("Iniezione dei pesi LoRA...")
model.load_adapter(LORA_MODEL_ID, adapter_name="flickr_lora")
model.eval()
print("[OK] Modello unificato pronto!")

# 4. Funzione di Generazione Unificata
def generate_vqa(model_type, image, question):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image.thumbnail((1024, 1024))
    image = image.convert("RGB")
    
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors='pt').to(DEVICE)
    
    # Switch degli adapter
    if model_type == "Base":
        model.disable_adapters()  # Modello originale
    else:
        model.enable_adapters()   # Modello Fine-Tuned
        
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=15, 
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id
        )
            
    new_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
    answer = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    answer = answer.split("ASSISTANT:")[-1].strip()
    answer = answer.split("USER:")[0].strip()
    return answer

# Wrapper per i bottoni
def predict_base(image, question):
    if image is None or not question: return "Input mancanti."
    return generate_vqa("Base", image, question)

def predict_tuned(image, question):
    if image is None or not question: return "Input mancanti."
    return generate_vqa("Fine-Tuned", image, question)

# 5. Interfaccia Gradio
with gr.Blocks(title='VQA Evaluation: Base vs QLoRA') as demo:
    gr.Markdown('# 📊 Analisi Comparativa VQA: LLaVA Base vs LLaVA Fine-Tuned (Flickr30k)')
    gr.Markdown(
        "Questa demo confronta dinamicamente l'attivazione e disattivazione dei pesi LoRA. "
        "A sinistra il modello senza gli adapter (Originale), a destra con gli adapter attivi."
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label='📷 Carica Immagine', type='pil')
            question_input = gr.Textbox(label='❓ Domanda (English)', placeholder='Es: What color is the sweater?')
            submit_btn = gr.Button('Interroga i Modelli 🚀', variant='primary')
            
        with gr.Column(scale=1):
            gr.Markdown("### 🟢 LLaVA Modello Base (Senza LoRA)")
            answer_base = gr.Textbox(label='Risposta Modello Base', lines=3)
            
        with gr.Column(scale=1):
            gr.Markdown("### 🔵 LLaVA Fine-Tuned (Con LoRA)")
            answer_tuned = gr.Textbox(label='Risposta Modello QLoRA', lines=3)
            
    submit_btn.click(fn=predict_base, inputs=[image_input, question_input], outputs=answer_base)
    submit_btn.click(fn=predict_tuned, inputs=[image_input, question_input], outputs=answer_tuned)
    
    question_input.submit(fn=predict_base, inputs=[image_input, question_input], outputs=answer_base)
    question_input.submit(fn=predict_tuned, inputs=[image_input, question_input], outputs=answer_tuned)

if __name__ == "__main__":
    demo.launch()