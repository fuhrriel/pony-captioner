#!/usr/bin/env python3
"""
Test script to verify GPU and CUDA support
"""

import sys

def test_cuda():
    print("=" * 60)
    print("Testing CUDA and GPU Support")
    print("=" * 60)
    
    # Test Python version
    print(f"✓ Python version: {sys.version.split()[0]}")
    
    # Test PyTorch
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"    Compute Capability: {props.major}.{props.minor}")
        else:
            print("✗ CUDA is not available!")
            print("  This likely means PyTorch was installed without CUDA support.")
            return False
    except ImportError as e:
        print(f"✗ Failed to import torch: {e}")
        return False
    
    # Test ONNX Runtime
    try:
        import onnxruntime as rt
        print(f"\n✓ ONNX Runtime version: {rt.__version__}")
        providers = rt.get_available_providers()
        print(f"✓ Available providers: {', '.join(providers)}")
        if 'CUDAExecutionProvider' in providers:
            print("✓ CUDA Execution Provider available")
        else:
            print("✗ CUDA Execution Provider not available")
    except ImportError as e:
        print(f"✗ Failed to import onnxruntime: {e}")
    
    # Test CLIP
    try:
        import clip
        print(f"\n✓ CLIP imported successfully")
        print(f"✓ Available CLIP models: {clip.available_models()}")
    except ImportError as e:
        print(f"✗ Failed to import CLIP: {e}")
    
    # Test other dependencies
    try:
        import lmdeploy
        print(f"\n✓ LMDeploy version: {lmdeploy.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import lmdeploy: {e}")
    
    try:
        from PIL import Image
        import numpy as np
        import pandas as pd
        import transformers
        print(f"✓ All core dependencies imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import dependencies: {e}")
        return False
    
    # Simple CUDA operation test
    if torch.cuda.is_available():
        try:
            print("\nTesting CUDA operations...")
            x = torch.randn(1000, 1000).cuda()
            y = torch.matmul(x, x.T)
            print("✓ CUDA tensor operations working")
        except Exception as e:
            print(f"✗ CUDA operation failed: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("All tests passed! System is ready.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_cuda()
    sys.exit(0 if success else 1)
