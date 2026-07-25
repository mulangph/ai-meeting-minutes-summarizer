# Meeting Minutes Generator (AI Engineering Portfolio)

An interactive, AI application that accepts audio inputs (microphone or file uploads), transcribes speech to text, and generates structured, production-ready meeting minutes in Markdown format.This project demonstrates practical competence in local model orchestration, hardware-aware memory optimization, and responsive interface design.

## 🚀 Key AI Engineering Highlights

This architecture implements several standard AI engineering patterns for local model deployment and resource management:

1. Open-Source LLM Inference : Instead of relying on hosted APIs, the application performs local inference
2. Automatic Speech Recognition (ASR) : Speech transcription is powered by whisper medium
3. Prompt Engineering : The application uses structured prompting with:
    -System Prompt
    -User Prompt
    -Chat Templates
4. Chat Template Formatting : The project uses: tokenizer.apply_chat_template(...) instead of manually building the prompts
5. 4-bit Quantization: This project uses BitsAndBytes 4-bit NF4 quantization to reduce VRAM usage while maintaining good generation quality.
6. GPU Memory Optimization : Running Whisper and Llama simultaneously can exceed available GPU memory.
This application intentionally executes them sequentially.
7. Dynamic Model Lifecycle : Instead of keeping every model permanently in memory, models are loaded only when required.
8. Hugging Face Pipelines : The project leverages Hugging Face's pipeline abstraction for ASR
9. Streaming User Experience : The application uses Python generators "yield" to provide progressive status updates.
10. Consumer GPU Deployment : The project demonstrates deploying modern open-source AI models without expensive enterprise hardware.

## 🛠️ System Requirements & Architecture
Hardware Dependencies

- GPU: NVIDIA Graphics Card with CUDA support (Minimum 6GB VRAM recommended).
- OS: Windows 10/11, macOS (requires adjustments for MPS), or Linux.

