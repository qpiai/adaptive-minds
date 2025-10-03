#!/usr/bin/env python3
"""
Adaptive Minds - Model Management CLI

Centralized tool to manage models, LoRAs, and configurations.

Usage:
    python manage_models.py sync          # Download/sync all models
    python manage_models.py list          # List all configured models
    python manage_models.py add-lora      # Register a new local LoRA
    python manage_models.py rebuild       # Rebuild metadata.json
    python manage_models.py validate      # Validate configuration
    python manage_models.py discover      # Discover unconfigured local LoRAs
"""

import argparse
import yaml
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG_FILE = "models_config.yaml"
METADATA_FILE = "metadata.json"

class ModelManager:
    def __init__(self, config_path: str = CONFIG_FILE):
        """Initialize model manager with configuration."""
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            logger.info("💡 Create models_config.yaml first or copy from models_config.example.yaml")
            sys.exit(1)
        
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Login to HF if token available
        hf_token = os.getenv('HF_TOKEN')
        if hf_token and hf_token != "your_huggingface_token_here":
            try:
                from huggingface_hub import login
                login(token=hf_token)
                logger.info("✅ Logged in to Hugging Face")
            except Exception as e:
                logger.warning(f"⚠️  Failed to login to HF: {e}")
        else:
            logger.warning("⚠️  No HF_TOKEN provided. Some models may require authentication.")
            logger.info("💡 Get your token from: https://huggingface.co/settings/tokens")
    
    def sync_models(self):
        """Download/sync all configured models."""
        logger.info("🔄 Syncing models from config...")
        
        # Sync base model
        base = self.config['base_model']
        if base['source'] == 'huggingface':
            self._download_model(base['huggingface_id'], base['local_path'], "base model")
        elif base['source'] == 'local':
            if os.path.exists(base['local_path']):
                logger.info(f"✅ Local base model found: {base['local_path']}")
            else:
                logger.error(f"❌ Local base model not found: {base['local_path']}")
        
        # Sync LoRA adapters
        enabled_adapters = [a for a in self.config['lora_adapters'] if a.get('enabled', True)]
        logger.info(f"📦 Found {len(enabled_adapters)} enabled LoRA adapters")
        
        for adapter in enabled_adapters:
            if adapter['source'] == 'huggingface':
                self._download_model(
                    adapter['huggingface_id'], 
                    adapter['local_path'],
                    f"LoRA: {adapter['name']}"
                )
            elif adapter['source'] == 'local':
                if os.path.exists(adapter['local_path']):
                    logger.info(f"✅ Local LoRA found: {adapter['name']}")
                else:
                    logger.warning(f"⚠️  Local LoRA missing: {adapter['name']} at {adapter['local_path']}")
        
        logger.info("\n✅ Sync complete!")
    
    def _download_model(self, model_id: str, local_path: str, model_type: str):
        """Download model from HuggingFace."""
        logger.info(f"📥 Downloading {model_type}: {model_id}")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model_id,
                local_dir=local_path,
                resume_download=True
            )
            logger.info(f"✅ Downloaded: {model_id}")
        except Exception as e:
            if "401" in str(e) or "gated" in str(e).lower() or "access" in str(e).lower():
                logger.error(f"❌ Access denied for {model_id}")
                logger.error(f"💡 This model requires authentication. Please:")
                logger.error(f"   1. Get a token from: https://huggingface.co/settings/tokens")
                logger.error(f"   2. Accept the license at: https://huggingface.co/{model_id}")
                logger.error(f"   3. Set HF_TOKEN environment variable")
            else:
                logger.error(f"❌ Failed to download {model_id}: {e}")
    
    def list_models(self):
        """List all configured models."""
        print("\n📊 Configured Models\n" + "="*60)
        
        # Base model
        base = self.config['base_model']
        print(f"\n🔹 Base Model: {base['name']}")
        print(f"   Source: {base['source']}")
        print(f"   Path: {base['local_path']}")
        if base['source'] == 'huggingface':
            print(f"   HF ID: {base['huggingface_id']}")
        
        # Check if exists
        exists = "📁" if os.path.exists(base['local_path']) else "❓"
        print(f"   Status: {exists}")
        
        # LoRA adapters
        adapters = self.config['lora_adapters']
        enabled_count = sum(1 for a in adapters if a.get('enabled', True))
        print(f"\n🔹 LoRA Adapters ({len(adapters)} total, {enabled_count} enabled):\n")
        
        for i, adapter in enumerate(adapters, 1):
            status = "✅" if adapter.get('enabled', True) else "❌"
            exists = "📁" if os.path.exists(adapter['local_path']) else "❓"
            print(f"{i}. {status} {exists} {adapter['name']}")
            print(f"   Source: {adapter['source']}")
            if adapter['source'] == 'huggingface':
                print(f"   HF ID: {adapter.get('huggingface_id', 'N/A')}")
            print(f"   Path: {adapter['local_path']}")
            print(f"   Description: {adapter['description']}")
            print()
    
    def rebuild_metadata(self):
        """Rebuild metadata.json from config."""
        logger.info("🔨 Rebuilding metadata.json...")
        
        # Build metadata structure
        metadata = {
            "base_model": {
                "name": self.config['base_model']['name'],
                "path": self.config['base_model']['local_path']
            },
            "router_prompt_template": self.config['router']['prompt_template'],
            "adapters": {},
            "_metadata": self.config['metadata']
        }
        
        # Add enabled adapters
        enabled_count = 0
        for adapter in self.config['lora_adapters']:
            if not adapter.get('enabled', True):
                continue
            
            metadata['adapters'][adapter['name']] = {
                "path": adapter['local_path'],
                "description": adapter['description'],
                "system_prompt": adapter['system_prompt'],
                "keywords": adapter['keywords']
            }
            enabled_count += 1
        
        # Write metadata.json
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Generated {METADATA_FILE} with {enabled_count} adapters")
    
    def add_local_lora(self, name: str, path: str, description: str, 
                       system_prompt: str, keywords: List[str]):
        """Add a new local LoRA to config."""
        logger.info(f"➕ Adding local LoRA: {name}")
        
        # Check if LoRA exists
        if not os.path.exists(path):
            logger.error(f"❌ LoRA not found at: {path}")
            return False
        
        # Check if adapter_config.json exists (indicates it's a LoRA)
        if not os.path.exists(os.path.join(path, "adapter_config.json")):
            logger.warning(f"⚠️  No adapter_config.json found. This might not be a LoRA directory.")
        
        new_adapter = {
            "name": name,
            "source": "local",
            "local_path": path,
            "description": description,
            "system_prompt": system_prompt,
            "keywords": keywords,
            "enabled": True
        }
        
        self.config['lora_adapters'].append(new_adapter)
        
        # Save config
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"✅ Added {name} to {CONFIG_FILE}")
        logger.info("💡 Run 'python manage_models.py rebuild' to update metadata.json")
        return True
    
    def validate_config(self):
        """Validate configuration and check file existence."""
        logger.info("🔍 Validating configuration...")
        issues = []
        
        # Check base model
        base = self.config['base_model']
        if base['source'] == 'local' and not os.path.exists(base['local_path']):
            issues.append(f"Base model not found: {base['local_path']}")
        
        # Check LoRAs
        for adapter in self.config['lora_adapters']:
            if not adapter.get('enabled', True):
                continue
            
            if adapter['source'] == 'local' and not os.path.exists(adapter['local_path']):
                issues.append(f"LoRA not found: {adapter['name']} at {adapter['local_path']}")
        
        if issues:
            logger.warning("⚠️  Issues found:")
            for issue in issues:
                logger.warning(f"   - {issue}")
            return False
        else:
            logger.info("✅ Configuration valid!")
            return True
    
    def discover_local_loras(self):
        """Auto-discover untrained LoRAs in loras/ directory."""
        logger.info("🔍 Discovering local LoRAs...")
        
        loras_dir = Path("./loras")
        if not loras_dir.exists():
            logger.warning("⚠️  ./loras directory not found")
            return
        
        # Get configured LoRAs
        configured_paths = {adapter['local_path'] for adapter in self.config['lora_adapters']}
        
        # Find LoRAs
        discovered = []
        for item in loras_dir.iterdir():
            if item.is_dir():
                # Check if it looks like a LoRA (has adapter_config.json)
                if (item / "adapter_config.json").exists():
                    lora_path = f"./loras/{item.name}"
                    if lora_path not in configured_paths:
                        discovered.append(item.name)
        
        if discovered:
            logger.info(f"\n📦 Found {len(discovered)} unconfigured LoRAs:")
            for name in discovered:
                logger.info(f"   - {name}")
            logger.info("\n💡 To add them, use:")
            logger.info("   python manage_models.py add-lora --name <name> --path ./loras/<name>")
        else:
            logger.info("✅ All local LoRAs are configured")
    
    def enable_adapter(self, name: str):
        """Enable a disabled adapter."""
        for adapter in self.config['lora_adapters']:
            if adapter['name'] == name:
                adapter['enabled'] = True
                logger.info(f"✅ Enabled adapter: {name}")
                
                # Save config
                with open(CONFIG_FILE, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
                return True
        
        logger.error(f"❌ Adapter not found: {name}")
        return False
    
    def disable_adapter(self, name: str):
        """Disable an adapter."""
        for adapter in self.config['lora_adapters']:
            if adapter['name'] == name:
                adapter['enabled'] = False
                logger.info(f"✅ Disabled adapter: {name}")
                
                # Save config
                with open(CONFIG_FILE, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
                return True
        
        logger.error(f"❌ Adapter not found: {name}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Adaptive Minds Model Management")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Sync command
    subparsers.add_parser('sync', help='Download/sync all models from config')
    
    # List command
    subparsers.add_parser('list', help='List all configured models')
    
    # Rebuild command
    subparsers.add_parser('rebuild', help='Rebuild metadata.json from config')
    
    # Validate command
    subparsers.add_parser('validate', help='Validate configuration')
    
    # Discover command
    subparsers.add_parser('discover', help='Discover unconfigured local LoRAs')
    
    # Add LoRA command
    add_parser = subparsers.add_parser('add-lora', help='Add a local LoRA to config')
    add_parser.add_argument('--name', required=True, help='LoRA name')
    add_parser.add_argument('--path', required=True, help='Local path to LoRA')
    add_parser.add_argument('--description', required=True, help='Description')
    add_parser.add_argument('--system-prompt', required=True, help='System prompt')
    add_parser.add_argument('--keywords', nargs='+', required=True, help='Keywords for routing')
    
    # Enable/Disable commands
    subparsers.add_parser('enable', help='Enable an adapter').add_argument('--name', required=True)
    subparsers.add_parser('disable', help='Disable an adapter').add_argument('--name', required=True)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = ModelManager()
    
    if args.command == 'sync':
        manager.sync_models()
    elif args.command == 'list':
        manager.list_models()
    elif args.command == 'rebuild':
        manager.rebuild_metadata()
    elif args.command == 'validate':
        manager.validate_config()
    elif args.command == 'discover':
        manager.discover_local_loras()
    elif args.command == 'add-lora':
        manager.add_local_lora(
            name=args.name,
            path=args.path,
            description=args.description,
            system_prompt=args.system_prompt,
            keywords=args.keywords
        )
    elif args.command == 'enable':
        manager.enable_adapter(args.name)
    elif args.command == 'disable':
        manager.disable_adapter(args.name)

if __name__ == "__main__":
    main()
