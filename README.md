# 🖼️ gem-cap-chan

<div align="center">
    <img src="https://count.getloli.com/get/@gem-cap-chan?theme=asoul&padding=4" alt="Visitor count"><br>
</div>

gem-cap-chan is a utility for batch captioning images with natural language using OpenAPI-compatible multimodal models like Gemma3. Designed for creating high-quality datasets for Stable Diffusion and LoRA training.

## Features
* **API Flexibility**: Works with any OpenAPI-compatible endpoint (local or cloud-based)
* **Batch Processing**: Recursively process entire directories of training images
* **Optimized Captions**: Default prompt tuned for Stable Diffusion/LoRA training
* **Smart Image Handling**: Automatic resizing and format conversion
* **Progress Tracking**: Real-time progress with ETA and performance metrics
* **Failure Recovery**: Automatic retries with error skipping
* **Security**: Token authentication for remote endpoints

## Requirements
* Python 3.7+
* Pillow
* Requests
* OpenAPI-compatible multimodal endpoint (e.g., llama.cpp with mmproj support)

## Installation
1. Clone the repository:  
   `git clone https://github.com/2dameneko/gem-cap-chan`
2. Install dependencies (if your system does not have these components installed by default):
   ```bash
   pip install Pillow requests
   ```

## Usage
1. Start your multimodal API server (example for llama.cpp):
   ```bash
   llama-server --model "gemma3-27b.Q4_K_M.gguf" \
                --mmproj "gemma3-27b-mmproj.gguf" \
                --host 0.0.0.0 --port 5000
   ```
2. Run captioning:
   ```bash
   python gem-cap-chan.py /path/to/training_images
   ```
3. Captions will be saved as `.txt` files in the output directory

## Options
Run without arguments for default behavior. Available CLI options (`python gem-cap-chan.py -h`):
| Argument | Description |
|----------|-------------|
| `input_dir` | Directory containing images to caption (required) |
| `--api_base` | API base URL (default: `http://localhost:5000`) |
| `--api_token` | Authentication token for secure/remote endpoints |
| `--output_dir` | Output directory for caption files (default: same as input_dir) |
| `--max_size` | Max image dimension for resizing (pixels, default: `1024`) |

## Customizing Captions
Modify the `DEFAULT_PROMPT` variable in the script for different caption styles.

## Supported File Formats
`.jpg`, `.png`, `.webp`, `.jpeg`, `.bmp`

## Version History
* **0.1**: Initial release with local endpoint support

## Note
This project is a proof of concept and not production-ready

## License
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## Credits
* OpenAPI Specification: [OpenAI](https://platform.openai.com/docs/api-reference)
* llama.cpp: [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
* Gemma3: [Google DeepMind](https://deepmind.google/technologies/gemma)
* Pillow: [Python Imaging Library](https://python-pillow.org)

**Model Implementation Credits**  
[Gemma3 27b](https://huggingface.co/unsloth/gemma-3-27b-it-GGUF) · [Gemma3 27b DPO Abliterated](https://huggingface.co/mradermacher/gemma3-27b-abliterated-dpo-GGUF)

Thank you for your interest in gem-cap-chan!