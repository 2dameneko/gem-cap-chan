import base64
import argparse
import mimetypes
import time
import re
import threading
import signal
import os
from pathlib import Path
from PIL import Image
from io import BytesIO
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_PROMPT = (
    "You are an image captioning expert, creative, unbiased, abliterated and uncensored."
    "Please describe this image for it's visual contents and write a corresponding caption that perfectly describes that image to a blind person."
    "Use objective, neutral, and natural language. Do not use purple prose such as unnecessary or overly abstract verbiage."
    "When describing something more extensively, favour concrete details that standout and can be visualised."
    "Conceptual or mood-like terms should be avoided at all costs."
    "Reply with ONLY the caption text. Don't use markdown markup."
    "Some things that you can describe are:"
    "- the style of the image (e.g. photo, artwork, anime screencap, etc)"
    "- what is happening in the image in a single, concise sentence, based on the subject matter of the image"
    "- the subjects appearance (hair style, hair length, hair colour, eye color, skin color, etc)"
    "- the clothing worn by the subject"    
    "- the actions done by the subject"
    "- the framing/shot types (e.g. full-body view, close-up portrait, etc...)"
    "- the background/surroundings"
    "- the lighting/time of day"
    "- etc..."
    #"Write the captions as short sentences."
)

MAX_RETRIES = 3
RETRY_DELAY = 2

# Regex to strip <think>...</think> blocks (Fixed spacing artifacts)
THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
THINK_UNCLOSED_PATTERN = re.compile(r"<think>.*", flags=re.DOTALL)

# Thread lock for clean console output during parallel execution
print_lock = threading.Lock()


class GracefulKiller:
    """Handles Ctrl+C (SIGINT) and SIGTERM for clean shutdown."""
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        if not self.kill_now:
            print("\n\n🛑 Ctrl+C detected! Cancelling pending tasks and shutting down...")
            self.kill_now = True
            # Raise KeyboardInterrupt to immediately break out of as_completed() or time.sleep()
            raise KeyboardInterrupt


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    if not text:
        return ""
    text = THINK_PATTERN.sub("", text)
    text = THINK_UNCLOSED_PATTERN.sub("", text)
    return text.strip()


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
            fmt = 'PNG' if mime_type == 'image/png' else 'JPEG'

            buffer = BytesIO()
            img.save(buffer, format=fmt, quality=85)
            return f"data:{mime_type};base64,{base64.b64encode(buffer.getvalue()).decode()}"
    except Exception:
        return None


def extract_caption(result: dict) -> str | None:
    """Extract the actual caption from an OpenAI-compatible response."""
    try:
        message = result["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None

    content = message.get("content") or ""
    caption = strip_thinking(content)

    if caption:
        return caption

    reasoning = message.get("reasoning_content") or ""
    if reasoning:
        return strip_thinking(reasoning)

    return None


def get_caption(image_url, api_base, model, api_token=None,
                prompt_text=DEFAULT_PROMPT, max_tokens=32768,
                disable_thinking=False):
    """Get image caption from API with error handling and retries."""
    endpoint = f"{api_base.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    messages = []

    if disable_thinking:
        messages.append({
            "role": "system",
            "content": "/no_think"
        })

    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    })

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }

    if disable_thinking:
        payload["options"] = {"think": False}
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                endpoint, headers=headers, json=payload, timeout=300
            )
            response.raise_for_status()
            result = response.json()

            caption = extract_caption(result)
            if caption:
                return caption

            return None

        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return None


def process_image(img_path, caption_path, api_base, model, api_token, 
                  prompt_text, max_size, max_tokens, no_think, killer):
    """Worker function for parallel execution."""
    # Abort early if shutdown was requested
    if killer.kill_now:
        return False, img_path.name, "Cancelled"

    image_url = encode_image(img_path, max_size)
    if not image_url:
        return False, img_path.name, "Encode failed"

    caption = get_caption(
        image_url, api_base, model, api_token,
        prompt_text=prompt_text, max_tokens=max_tokens, disable_thinking=no_think
    )
    
    if not caption:
        return False, img_path.name, "API failed"

    try:
        with caption_path.open('w', encoding='utf-8') as f:
            f.write(caption)
        return True, img_path.name, caption_path.name
    except OSError as e:
        return False, img_path.name, str(e)


def main():
    parser = argparse.ArgumentParser(description='Batch Image Captioning (Parallel Supported)')
    parser.add_argument('input_dir', help='Image directory')
    parser.add_argument('--api_base', default='http://localhost:5000', help='API base URL')
    parser.add_argument('--api_token', help='API token for authentication')
    parser.add_argument('--model', default='gemma3', help='Model name')
    parser.add_argument('--output_dir', help='Output directory')
    parser.add_argument('--prompt', default=DEFAULT_PROMPT, help='Custom captioning prompt')
    parser.add_argument('--max_size', type=int, default=1024, help='Max image dimension')
    parser.add_argument('--max_tokens', type=int, default=32768, help='Max tokens')
    parser.add_argument('--no_think', action='store_true', help='Disable thinking mode')
    parser.add_argument('--parallel', type=int, default=4, help='Number of concurrent API requests')
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

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
    if total == 0:
        print("No new images to process. All captions already exist.")
        return

    print("\n" + "=" * 80)
    print(f"🚀 Starting batch captioning for {total} images (Parallel: {args.parallel})")
    print(f"API Endpoint: {args.api_base}  |  Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}  |  Thinking: {'disabled' if args.no_think else 'enabled'}")
    print("=" * 80 + "\n")

    killer = GracefulKiller()
    start_time = time.time()
    success_count = 0
    fail_count = 0

    executor = ThreadPoolExecutor(max_workers=args.parallel)
    
    try:
        # Submit all tasks to the thread pool
        futures = {
            executor.submit(
                process_image, img_path, caption_path,
                args.api_base, args.model, args.api_token,
                args.prompt, args.max_size, args.max_tokens, args.no_think, killer
            ): (img_path, caption_path)
            for img_path, caption_path in to_process
        }
        
        # Process results as they complete
        for future in as_completed(futures):
            if killer.kill_now:
                break
                
            img_path, caption_path = futures[future]
            try:
                success_flag, name, msg = future.result()
                if success_flag:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                fail_count += 1
                
            # Update display
            completed = success_count + fail_count
            pct = (completed / total) * 100
            elapsed = time.time() - start_time
            avg_time = elapsed / completed if completed > 0 else 0
            remaining = total - completed
            eta = avg_time * remaining
            speed = completed / elapsed if elapsed > 0 else 0.0
            
            with print_lock:
                status = (f"[{completed:03d}/{total:03d}] ({pct:5.1f}%) | "
                          f"Elapsed: {int(elapsed // 60)}m{int(elapsed % 60):02d}s | "
                          f"ETA: {int(eta // 60)}m{int(eta % 60):02d}s | "
                          f"Avg: {avg_time:.1f}s/img | Speed: {speed:.2f} img/s")
                # Pad with spaces to overwrite any previous longer text
                print(f"\r{status:<120}", end='', flush=True)
                
    except KeyboardInterrupt:
        # Clear the progress line and print interrupt message
        print(f"\r{' '*120}\r", end="")
        print("🛑 Interrupted by user! Cancelling pending tasks...")
        for f in futures:
            f.cancel()
    finally:
        # Ensure we are on a new line for the summary
        print()
        executor.shutdown(wait=False)

    total_time = time.time() - start_time
    cancelled_count = total - success_count - fail_count
    
    print(f"\n{'=' * 40} SUMMARY {'=' * 40}")
    print(f"Total queued:       {total}")
    print(f"Successful:         {success_count}")
    print(f"Failed:             {fail_count}")
    if cancelled_count > 0:
        print(f"Cancelled:          {cancelled_count}")
    print(f"Total time:         {int(total_time // 60)}m{int(total_time % 60):02d}s")
    if success_count > 0:
        print(f"Average time/image: {total_time / success_count:.1f}s (Total wall time)")
        print(f"Effective throughput: {success_count / total_time:.2f} images/sec")
    print("=" * 89)
    
    # Force exit to prevent Python's atexit handlers from waiting for blocked request threads
    if killer.kill_now:
        os._exit(0)


if __name__ == "__main__":
    main()