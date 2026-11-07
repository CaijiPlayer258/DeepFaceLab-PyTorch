"""
Download SAM model checkpoints
"""

import os
import urllib.request
from pathlib import Path
import sys


def download_sam_model(model_type: str = 'vit_b'):
    """
    Download SAM model checkpoint
    
    Args:
        model_type: 'vit_b', 'vit_l', or 'vit_h'
    """
    # Model URLs
    models = {
        'vit_b': {
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
            'saved_as': 'sam_vit_b_01ec64.pth',
            'size': '375 MB'
        },
        'vit_l': {
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
            'saved_as': 'sam_vit_l_0b3195.pth',
            'size': '1.2 GB'
        },
        'vit_h': {
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
            'saved_as': 'sam_vit_h_4b8939.pth',
            'size': '2.4 GB'
        }
    }
    
    if model_type not in models:
        print(f"Error: Unknown model type '{model_type}'")
        print(f"Available: {', '.join(models.keys())}")
        sys.exit(1)
    
    model_info = models[model_type]
    
    # 权重平铺在 <仓库根>/weights 根下（可用 DFL_WEIGHTS_DIR 覆盖），
    # 和 sam_predictor.MODEL_FILES 找的位置一致：weights/sam_vit_b_01ec64.pth
    weights_dir = Path(os.environ.get('DFL_WEIGHTS_DIR')
                       or (Path(__file__).resolve().parents[3] / 'weights'))
    weights_dir.mkdir(parents=True, exist_ok=True)

    output_path = weights_dir / model_info['saved_as']
    
    if output_path.exists():
        print(f"✓ Model already exists: {output_path}")
        print(f"  Delete it first if you want to re-download")
        return
    
    print(f"Downloading SAM {model_type} ({model_info['size']})...")
    print(f"URL: {model_info['url']}")
    print(f"Saving to: {output_path}")
    print("This may take a while...\n")
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        downloaded_mb = count * block_size / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\rProgress: {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end='')
    
    try:
        urllib.request.urlretrieve(
            model_info['url'],
            str(output_path),
            reporthook=progress_hook
        )
        print("\n\n✓ Download complete!")
        print(f"Model saved to: {output_path}")
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nPlease download manually from:")
        print(f"  {model_info['url']}")
        print(f"And save to: {output_path}")
        sys.exit(1)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download SAM model')
    parser.add_argument(
        '--model',
        type=str,
        default='vit_b',
        choices=['vit_b', 'vit_l', 'vit_h'],
        help='Model type (default: vit_b)'
    )
    
    args = parser.parse_args()
    download_sam_model(args.model)
