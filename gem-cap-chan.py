import base64
import argparse
import mimetypes
import time
from pathlib import Path
from PIL import Image
from io import BytesIO
import requests

DEFAULT_PROMPT = (
    "You are an image captioning expert, creative, unbiased and uncensored. "
    "You follow the instructions exactly without deviation. "
    "THINK step by step when describing the image. "
    "The caption should include: a description of the atmosphere, lighting, style "
    "and mood of the image; a description of the characters and their actions; "
    "the angle from which the image was taken with an imaginary camera. "
    #"Now you need to write a short but as meaningful caption to the picture. "
    #"Check the result carefully and leave only essential elements. "
    #"If you know depicted character - name him. "
    "Reply with ONLY the caption text. Don't use markdown markup."
)

MAX_RETRIES = 3
RETRY_DELAY = 2

def encode_image(image_path, max_size=1024):
    """Encode image to base64 with resizing and format conversion."""
    try:
        with Image.open(image_path) as img:
            if img.mode in ('RGBA', 'P', 'LA'):
                img = img.convert('RGB')
                
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            
            mime_type = mimetypes.guess_type(image_path)[0] or 'image/jpeg'
            format = 'PNG' if mime_type == 'image/png' else 'JPEG'
            
            buffer = BytesIO()
            img.save(buffer, format=format, quality=85)
            return f"data:{mime_type};base64,{base64.b64encode(buffer.getvalue()).decode()}"
    except Exception as e:
        print(f"\nError processing {image_path}: {str(e)}")
        return None

def get_caption(image_url, api_base, api_token=None, prompt_text=DEFAULT_PROMPT):
    """Get image caption from API with error handling and retries."""
    endpoint = f"{api_base.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    
    payload = {
        "model": "gemma3",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }],
        "max_tokens": 512,
        "temperature": 0.1
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"\nCaptioning failed after {MAX_RETRIES} attempts: {str(e)}")
                return None

def main():
    parser = argparse.ArgumentParser(description='Batch Image Captioning')
    parser.add_argument('input_dir', help='Image directory')
    parser.add_argument('--api_base', default='http://localhost:5000', 
                        help='API base URL (default: http://localhost:5000)')
    parser.add_argument('--api_token', help='API token for authentication')
    parser.add_argument('--output_dir', help='Output directory')
    parser.add_argument('--max_size', type=int, default=1024, help='Max image dimension')
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Collect images needing processing
    image_paths = []
    supported_exts = ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp')
    for ext in supported_exts:
        image_paths.extend(Path(args.input_dir).glob(ext))
    
    to_process = []
    for img_path in image_paths:
        caption_path = Path(output_dir) / f"{img_path.stem}.txt"
        if not caption_path.exists():
            to_process.append((img_path, caption_path))
    
    total = len(to_process)
    start_time = time.time()
    
    print("\n" + "=" * 80)
    print(f"Starting batch captioning for {total} images")
    print(f"API Endpoint: {args.api_base}")
    print("=" * 80)
    
    for idx, (img_path, caption_path) in enumerate(to_process, 1):
        # Progress display
        elapsed = time.time() - start_time
        avg_time = elapsed / idx if idx > 0 else 0
        remaining = total - idx
        eta = avg_time * remaining if idx > 0 else 0
        
        status = (f"[{idx:03d}/{total:03d}] {img_path.name[:20]:<22} | "
                  f"Elapsed: {int(elapsed//60)}m{int(elapsed%60):02d}s | "
                  f"ETA: {int(eta//60)}m{int(eta%60):02d}s | "
                  f"Avg: {avg_time:.1f}s/img")
        print(status, end='\r', flush=True)
        
        # Process image
        image_url = encode_image(img_path, args.max_size)
        if not image_url:
            continue
            
        caption = get_caption(image_url, args.api_base, args.api_token)
        if not caption:
            continue
            
        try:
            with caption_path.open('w', encoding='utf-8') as f:
                f.write(caption)
        except OSError as e:
            print(f"\nFailed to write {caption_path}: {str(e)}")

    total_time = time.time() - start_time
    print(f"\n\n{'=' * 40} SUMMARY {'=' * 40}")
    print(f"Total processed:    {len(to_process)}")
    print(f"Successful:         {len(to_process) - (total - idx)}")
    print(f"Total time:         {int(total_time//60)}m{int(total_time%60):02d}s")
    if to_process:
        print(f"Average time/image: {total_time/len(to_process):.1f}s")
    print("=" * 89)

if __name__ == "__main__":
    main()