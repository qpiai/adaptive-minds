#!/usr/bin/env python3
"""
Test script for the generalized Adaptive Minds workflow.
This validates that all components work together correctly.
"""

import os
import sys
import subprocess
import yaml
import json
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n🔧 {description}")
    print(f"   Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Success")
            return True
        else:
            print(f"   ❌ Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_file_exists(path, description):
    """Check if a file exists."""
    if os.path.exists(path):
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description}: {path} (not found)")
        return False

def validate_yaml_config():
    """Validate the YAML configuration."""
    print("\n📋 Validating models_config.yaml...")
    
    if not os.path.exists("models_config.yaml"):
        print("❌ models_config.yaml not found")
        return False
    
    try:
        with open("models_config.yaml") as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['base_model', 'lora_adapters', 'router', 'metadata']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing section: {section}")
                return False
        
        # Check base model
        base = config['base_model']
        if 'name' not in base or 'source' not in base:
            print("❌ Base model missing required fields")
            return False
        
        # Check adapters
        adapters = config['lora_adapters']
        if not isinstance(adapters, list):
            print("❌ lora_adapters should be a list")
            return False
        
        print(f"✅ Configuration valid ({len(adapters)} adapters configured)")
        return True
        
    except Exception as e:
        print(f"❌ YAML validation failed: {e}")
        return False

def test_manage_models():
    """Test the manage_models.py CLI."""
    print("\n🛠️  Testing manage_models.py...")
    
    # Test help
    if not run_command("python manage_models.py --help", "Testing help command"):
        return False
    
    # Test list command
    if not run_command("python manage_models.py list", "Testing list command"):
        return False
    
    # Test validate command
    if not run_command("python manage_models.py validate", "Testing validate command"):
        return False
    
    # Test discover command
    if not run_command("python manage_models.py discover", "Testing discover command"):
        return False
    
    print("✅ manage_models.py CLI working correctly")
    return True

def test_metadata_generation():
    """Test metadata.json generation."""
    print("\n📄 Testing metadata.json generation...")
    
    # Rebuild metadata
    if not run_command("python manage_models.py rebuild", "Rebuilding metadata.json"):
        return False
    
    # Check if metadata.json was created
    if not check_file_exists("metadata.json", "Generated metadata.json"):
        return False
    
    # Validate JSON structure
    try:
        with open("metadata.json") as f:
            metadata = json.load(f)
        
        required_keys = ['router_prompt_template', 'adapters', '_metadata']
        for key in required_keys:
            if key not in metadata:
                print(f"❌ metadata.json missing key: {key}")
                return False
        
        print(f"✅ metadata.json valid ({len(metadata['adapters'])} adapters)")
        return True
        
    except Exception as e:
        print(f"❌ JSON validation failed: {e}")
        return False

def test_training_script():
    """Test the training script."""
    print("\n🎓 Testing training script...")
    
    # Check if training script exists
    if not check_file_exists("train/train.py", "Training script"):
        return False
    
    # Test help
    if not run_command("python train/train.py --help", "Testing training script help"):
        return False
    
    print("✅ Training script accessible")
    return True

def test_server_components():
    """Test server components."""
    print("\n🌐 Testing server components...")
    
    # Check server.py
    if not check_file_exists("server.py", "Main server"):
        return False
    
    # Check if server can import dependencies
    if not run_command("python -c 'import server; print(\"Server imports OK\")'", "Testing server imports"):
        return False
    
    print("✅ Server components ready")
    return True

def main():
    """Run all tests."""
    print("🧪 Adaptive Minds Workflow Test")
    print("=" * 50)
    
    # Change to playground directory
    if not os.path.exists("models_config.yaml"):
        print("❌ Not in playground directory or models_config.yaml missing")
        print("💡 Run this from the playground directory")
        return False
    
    tests = [
        ("YAML Configuration", validate_yaml_config),
        ("Model Management CLI", test_manage_models),
        ("Metadata Generation", test_metadata_generation),
        ("Training Script", test_training_script),
        ("Server Components", test_server_components),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Results Summary")
    print(f"{'='*50}")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your workflow is ready to use.")
        print("\n💡 Next steps:")
        print("   1. python manage_models.py sync    # Download models")
        print("   2. python server.py               # Start server")
        print("   3. python train/train.py --help   # Train custom LoRAs")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Check the errors above.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
