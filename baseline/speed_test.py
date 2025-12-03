from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import time

def load_model(model_name="Qwen/Qwen2.5-7B-Instruct"):
    """Load model and tokenizer"""
    print(f"Loading {model_name}...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print("Loading model with SDPA attention...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa"  # This is already optimized
    )
    print(f"Model loaded successfully!")
    print(f"VRAM used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    return model, tokenizer

def run_inference(model, tokenizer, prompts):
    """Run multiple inference requests and measure time"""
    results = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*60}")
        print(f"Request {i}/{len(prompts)}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")
        
        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Generate with timing
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        elapsed = time.time() - start
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
        
        # Print results
        print(f"\nResponse: {response[len(prompt):]}")
        print(f"\n--- Performance ---")
        print(f"Time taken: {elapsed:.2f}s")
        print(f"Tokens generated: {tokens_generated}")
        print(f"Speed: {tokens_generated/elapsed:.1f} tokens/sec")
        print(f"VRAM used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        
        results.append({
            'prompt': prompt,
            'time': elapsed,
            'tokens': tokens_generated,
            'speed': tokens_generated/elapsed
        })
    
    return results

if __name__ == "__main__":
    # Test prompts
    prompts = [
        "Explain quantum computing in simple terms.",
        "What are the key differences between supervised and unsupervised learning?",
        "Write a Python function to calculate fibonacci numbers.",
        "What is gradient descent and how does it work?"
    ]
    
    # Load model
    model, tokenizer = load_model()
    
    # Run inference
    print("\n" + "="*60)
    print("STARTING INFERENCE TESTS")
    print("="*60)
    
    results = run_inference(model, tokenizer, prompts)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    avg_time = sum(r['time'] for r in results) / len(results)
    avg_speed = sum(r['speed'] for r in results) / len(results)
    total_time = sum(r['time'] for r in results)
    
    print(f"Total requests: {len(results)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per request: {avg_time:.2f}s")
    print(f"Average speed: {avg_speed:.1f} tokens/sec")
