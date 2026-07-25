import gradio as gr
import os
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig, pipeline
import torch
from dotenv import load_dotenv

load_dotenv()

hf_token = os.environ.get("HF_TOKEN")
login(hf_token, add_to_git_credential=True)

# Constants
LLAMA = "meta-llama/Llama-3.2-3B-Instruct"

def summarize_in_markup(transcription):
    system_message = """
    You produce minutes of meetings from transcripts, with summary, key discussion points,
    takeaways and action items with owners, in markdown format without code blocks.
    """

    user_prompt = f"""
    Below is an extract transcript of a a meeting.
    Please write minutes in markdown without code blocks, including:
    - a summary with attendees, location and date
    - discussion points
    - takeaways
    - action items with owners

    Transcription:
    {transcription}
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]

    # Quantization is configured here inside the execution thread
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(LLAMA)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
    
    streamer = TextStreamer(tokenizer)
    
    # Model loads dynamically, processes, and releases
    model = AutoModelForCausalLM.from_pretrained(LLAMA, device_map="auto", quantization_config=quant_config)
    outputs = model.generate(**inputs, max_new_tokens=2000, streamer=streamer)
    
    final_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Explicitly clear GPU memory tracking references
    del model
    del inputs
    torch.cuda.empty_cache()
    
    return final_text


def process_audio(audio_file):
    if audio_file is None:
        yield "❌ Error: No audio file provided!"
        return

    yield "⏳ *Processing audio file and executing Whisper speech-to-text... Please wait.*"
    
    # Whisper loads, processes, and finishes here sequentially
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-medium.en",
        dtype=torch.float16,
        device='cuda',
        return_timestamps=True
    )
    
    result = pipe(audio_file)
    transcription = result["text"]
    
    # Clean up Whisper from VRAM before jumping into Llama
    del pipe
    torch.cuda.empty_cache()

    yield "🤖 *Audio successfully transcribed! The Llama model is now compiling your structured meeting minutes...*"

    # Generate the text string and explicitly yield it to the UI
    final_markdown = summarize_in_markup(transcription)
    yield final_markdown


# Gradio Interface Setup
interface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath"),
    outputs=gr.Markdown(),
    title="Meeting Minutes Generator",
    description="Upload or record an audio file to get structured meeting minutes in Markdown format instantly.",
    show_progress="full"
)

if __name__ == "__main__":
    interface.launch(inbrowser=True)
