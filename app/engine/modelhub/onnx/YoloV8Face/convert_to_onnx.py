"""
Convert YOLOv8 Face model from PyTorch (.pt) to ONNX format
"""
import sys
from pathlib import Path

# Add project root to path
# This script is in: modelhub/onnx/YoloV8Face/convert_to_onnx.py
# So we need to go up 3 levels to reach project root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def convert_yolov8_to_onnx():
    """Convert YOLOv8 .pt model to ONNX format"""
    
    # Model paths
    pt_model_path = project_root / "modelhub" / "onnx" / "YoloV8Face" / "yolov8n-face.pt"
    onnx_model_path = project_root / "modelhub" / "onnx" / "YoloV8Face" / "yolov8n-face.onnx"
    
    print("="*80)
    print("YOLOv8 Face Model Converter: PT -> ONNX")
    print("="*80)
    print(f"\nInput:  {pt_model_path}")
    print(f"Output: {onnx_model_path}\n")
    
    # Check if input file exists
    if not pt_model_path.exists():
        print(f"ERROR: Input file not found: {pt_model_path}")
        return False
    
    try:
        # Try to import ultralytics
        from ultralytics import YOLO
        
        print("Loading YOLOv8 model...")
        model = YOLO(str(pt_model_path))
        
        print("Exporting to ONNX format...")
        # Export to ONNX with optimizations
        model.export(
            format='onnx',
            opset=12,           # ONNX opset version
            dynamic=False,      # Fixed input size (better for inference)
            simplify=True,      # Simplify the model
            imgsz=640,          # Input image size
        )
        
        # The exported file will be in the same directory with .onnx extension
        # Move it to the desired location if needed
        exported_onnx = pt_model_path.with_suffix('.onnx')
        
        if exported_onnx.exists():
            # Copy to target location if different
            if exported_onnx != onnx_model_path:
                import shutil
                shutil.copy2(str(exported_onnx), str(onnx_model_path))
                print(f"\n✓ Model copied to: {onnx_model_path}")
            else:
                print(f"\n✓ Model exported to: {onnx_model_path}")
            
            # Verify the output
            import os
            file_size = os.path.getsize(onnx_model_path)
            file_size_mb = file_size / (1024 * 1024)
            print(f"✓ File size: {file_size_mb:.2f} MB")
            print("\nConversion completed successfully!")
            return True
        else:
            print(f"ERROR: ONNX file not created at expected location")
            return False
            
    except ImportError:
        print("ERROR: ultralytics package not found!")
        print("\nPlease install it with:")
        print("  pip install ultralytics")
        return False
        
    except Exception as e:
        print(f"\nERROR during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = convert_yolov8_to_onnx()
    sys.exit(0 if success else 1)
