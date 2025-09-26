#!/usr/bin/env python3
"""
Hugging Face Model Downloader for Base Models and LoRA Files

This script downloads base models and LoRA files from Hugging Face Hub.
It supports downloading multiple models and provides flexible configuration options.

Usage:
    python pull_models_hf.py --type base --model llama-3.1-8B-instruct
    python pull_models_hf.py --type lora --model llama-8B-medical-alpaca
    python pull_models_hf.py --type all
    python pull_models_hf.py --type lora --all-loras
"""

import os
import logging
import sys
import argparse
from huggingface_hub import snapshot_download, login
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Base directory for Docker container
BASE_DIR = "/app"

# Hugging Face Configuration
HF_TOKEN = os.getenv('HF_TOKEN', None)  # Set this for private models

# Model Configurations with Hugging Face model IDs
MODEL_CONFIGS = {
    'base': {
        'local_path': os.getenv('BASE_MODEL_LOCAL_PATH', os.path.join(BASE_DIR, "base_model")),
        'models': {
            'llama-3.1-8B-instruct': {
                'model_id': 'meta-llama/Llama-3.1-8B-Instruct',
                'local_path': os.path.join(BASE_DIR, "base_model", "llama-3.1-8B-instruct")
            }
        }
    },
    'lora': {
        'local_path': os.getenv('LORA_MODEL_LOCAL_PATH', os.path.join(BASE_DIR, "loras")),
        'models': {
            'llama-8B-alpaca-2k': {
                'model_id': 'pavan01729/llama-8B-alpaca-2k',  # Update with your actual HF model ID
                'local_path': os.path.join(BASE_DIR, "loras", "llama-8B-alpaca-2k")
            },
            'llama-8B-chemistry': {
                'model_id': 'pavan01729/llama-8B-chemistry',  # Update with your actual HF model ID
                'local_path': os.path.join(BASE_DIR, "loras", "llama-8B-chemistry")
            },
            'llama-8B-finance-alpaca': {
                'model_id': 'pavan01729/llama-8B-finance-alpaca',  # Update with your actual HF model ID
                'local_path': os.path.join(BASE_DIR, "loras", "llama-8B-finance-alpaca")
            },
            'llama-8B-gpt-ai': {
                'model_id': 'pavan01729/llama-8B-gpt-ai',
                'local_path': os.path.join(BASE_DIR, "loras", "llama-8B-gpt-ai")
            },
            'llama-8B-medical-alpaca': {
                'model_id': 'pavan01729/llama-8B-medical-alpaca',
                'local_path': os.path.join(BASE_DIR, "loras", "llama-8B-medical-alpaca")
            }
        }
    }
}

class HuggingFaceModelDownloader:
    def __init__(self):
        if HF_TOKEN and HF_TOKEN != "your_huggingface_token_here":
            try:
                login(token=HF_TOKEN)
                logger.info("✅ Logged in to Hugging Face with token")
            except Exception as e:
                logger.error(f"❌ Failed to login to Hugging Face: {e}")
                raise
        else:
            logger.warning("⚠️ No HF_TOKEN provided. Some models may require authentication.")
            logger.info("💡 To access gated models like Llama 3.1, set HF_TOKEN environment variable")
            logger.info("💡 Get your token from: https://huggingface.co/settings/tokens")

    def download_model(self, model_type: str, model_name: str) -> bool:
        """
        Download a specific model from Hugging Face Hub.
        
        Args:
            model_type: 'base' or 'lora'
            model_name: Name of the model to download
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if model_type not in MODEL_CONFIGS:
                logger.error(f"Invalid model type: {model_type}")
                return False
                
            config = MODEL_CONFIGS[model_type]
            if model_name not in config['models']:
                logger.error(f"Model '{model_name}' not found in {model_type} models")
                return False
                
            model_config = config['models'][model_name]
            model_id = model_config['model_id']
            local_path = model_config['local_path']
            
            logger.info(f"Starting download of {model_type} model '{model_name}' from Hugging Face")
            logger.info(f"Model ID: {model_id}")
            logger.info(f"Local path: {local_path}")
            
            # Create local directory if it doesn't exist
            os.makedirs(local_path, exist_ok=True)
            
            # Download the model
            try:
                snapshot_download(
                    repo_id=model_id,
                    local_dir=local_path,
                    resume_download=True  # Resume if interrupted
                )
            except Exception as e:
                if "401" in str(e) or "gated" in str(e).lower() or "access" in str(e).lower():
                    logger.error(f"❌ Access denied for {model_id}")
                    logger.error(f"💡 This model requires authentication. Please:")
                    logger.error(f"   1. Get a token from: https://huggingface.co/settings/tokens")
                    logger.error(f"   2. Accept the license at: https://huggingface.co/{model_id}")
                    logger.error(f"   3. Set HF_TOKEN environment variable")
                    return False
                else:
                    raise
            
            logger.info(f"Download completed for {model_name}. Files saved to: {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {model_type} model '{model_name}': {e}")
            return False

    def download_all_models(self, model_type: str) -> Dict[str, bool]:
        """
        Download all models of a specific type.
        
        Args:
            model_type: 'base' or 'lora'
            
        Returns:
            Dict[str, bool]: Results for each model
        """
        if model_type not in MODEL_CONFIGS:
            logger.error(f"Invalid model type: {model_type}")
            return {}
            
        config = MODEL_CONFIGS[model_type]
        results = {}
        
        logger.info(f"Starting download of all {model_type} models...")
        
        for model_name in config['models']:
            logger.info(f"Processing {model_type} model: {model_name}")
            results[model_name] = self.download_model(model_type, model_name)
            
        return results

    def download_all(self) -> Dict[str, Dict[str, bool]]:
        """
        Download all base models and LoRA files.
        
        Returns:
            Dict[str, Dict[str, bool]]: Results for all model types
        """
        logger.info("Starting download of all models from Hugging Face...")
        
        results = {}
        results['base'] = self.download_all_models('base')
        results['lora'] = self.download_all_models('lora')
        
        return results

def print_summary(results: Dict[str, bool], model_type: str):
    """Print a summary of download results."""
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    logger.info(f"\n=== {model_type.upper()} MODELS DOWNLOAD SUMMARY ===")
    logger.info(f"Successfully downloaded: {successful}/{total}")
    
    for model_name, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {model_name}")
    
    if successful < total:
        logger.warning(f"Failed to download {total - successful} {model_type} model(s)")

def main():
    parser = argparse.ArgumentParser(description="Download base models and LoRA files from Hugging Face Hub")
    parser.add_argument('--type', choices=['base', 'lora', 'all'], 
                       help='Type of models to download')
    parser.add_argument('--model', type=str,
                       help='Specific model name to download')
    parser.add_argument('--all-loras', action='store_true',
                       help='Download all LoRA models')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models')
    
    args = parser.parse_args()
    
    # List available models if requested
    if args.list_models:
        logger.info("=== AVAILABLE MODELS ===")
        for model_type, config in MODEL_CONFIGS.items():
            logger.info(f"\n{model_type.upper()} MODELS:")
            for model_name, model_config in config['models'].items():
                model_id = model_config['model_id']
                local_path = model_config['local_path']
                logger.info(f"  - {model_name} ({model_id}) -> {local_path}")
        return
    
    # Default behavior: download all models if no arguments provided
    if not args.type and not args.model and not args.all_loras:
        logger.info("No arguments provided. Downloading all models (base + LoRAs) from Hugging Face...")
        downloader = HuggingFaceModelDownloader()
        try:
            results = downloader.download_all()
            print_summary(results['base'], 'base')
            print_summary(results['lora'], 'lora')
        except KeyboardInterrupt:
            logger.info("Download interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        return
    
    downloader = HuggingFaceModelDownloader()
    
    try:
        if args.type == 'all':
            # Download all models
            results = downloader.download_all()
            print_summary(results['base'], 'base')
            print_summary(results['lora'], 'lora')
            
        elif args.type == 'base':
            if args.model:
                # Download specific base model
                success = downloader.download_model('base', args.model)
                if success:
                    logger.info(f"Successfully downloaded base model: {args.model}")
                else:
                    logger.error(f"Failed to download base model: {args.model}")
            else:
                # Download all base models
                results = downloader.download_all_models('base')
                print_summary(results, 'base')
                
        elif args.type == 'lora':
            if args.model:
                # Download specific LoRA model
                success = downloader.download_model('lora', args.model)
                if success:
                    logger.info(f"Successfully downloaded LoRA model: {args.model}")
                else:
                    logger.error(f"Failed to download LoRA model: {args.model}")
            else:
                # Download all LoRA models
                results = downloader.download_all_models('lora')
                print_summary(results, 'lora')
                
        elif args.all_loras:
            # Download all LoRA models
            results = downloader.download_all_models('lora')
            print_summary(results, 'lora')
            
        elif args.model:
            # Try to determine model type and download
            model_found = False
            for model_type, config in MODEL_CONFIGS.items():
                if args.model in config['models']:
                    success = downloader.download_model(model_type, args.model)
                    if success:
                        logger.info(f"Successfully downloaded {model_type} model: {args.model}")
                    else:
                        logger.error(f"Failed to download {model_type} model: {args.model}")
                    model_found = True
                    break
            
            if not model_found:
                logger.error(f"Model '{args.model}' not found in any model type")
                
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()

