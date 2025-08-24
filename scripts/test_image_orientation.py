#!/usr/bin/env python3
"""
Test Script: Check Image Orientation
Shows the current image orientation and all possible rotations.
"""

import os
import sys
import numpy as np
from PIL import Image
import argparse

def test_image_orientation(image_path: str, output_dir: str):
    """
    Test different image orientations and save them for comparison.
    
    Args:
        image_path: Path to the PNG image
        output_dir: Directory to save rotated images
    """
    print(f"🔍 Testing image orientation: {os.path.basename(image_path)}")
    
    # Load image
    img_array = np.array(Image.open(image_path))
    
    # Check original dimensions
    height, width = img_array.shape[:2]
    print(f"📐 Original image dimensions: {width} × {height}")
    
    # Determine if image is portrait or landscape
    if height > width:
        print("📱 Image is PORTRAIT (taller than wide)")
    elif width > height:
        print("🖼️ Image is LANDSCAPE (wider than tall)")
    else:
        print("⬜ Image is SQUARE")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Try different rotations
    rotation_options = [
        ("original", img_array, "No rotation", "Original orientation"),
        ("clockwise_90", np.rot90(img_array, k=-1), "90° clockwise", "Rotated 90° clockwise"),
        ("counter_clockwise_90", np.rot90(img_array, k=1), "90° counter-clockwise", "Rotated 90° counter-clockwise"),
        ("180", np.rot90(img_array, k=2), "180° rotation", "Rotated 180°")
    ]
    
    print(f"\n🔄 Testing different rotations:")
    print("-" * 50)
    
    for rotation_name, rotated_img, rotation_desc, rotation_info in rotation_options:
        h, w = rotated_img.shape[:2]
        print(f"   {rotation_desc}: {w} × {h}")
        
        # Save rotated image
        rotation_path = os.path.join(output_dir, f"rotation_{rotation_name}.png")
        Image.fromarray(rotated_img).save(rotation_path)
        print(f"      💾 Saved: {rotation_path}")
        
        # Determine orientation after rotation
        if h > w:
            orientation = "PORTRAIT"
        elif w > h:
            orientation = "LANDSCAPE"
        else:
            orientation = "SQUARE"
        
        print(f"      📱 Orientation: {orientation}")
        print(f"      ℹ️ {rotation_info}")
        print()
    
    print(f"🎯 All rotation options saved to: {output_dir}")
    print(f"💡 Check the images to see which orientation looks correct for your table!")

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Test different image orientations')
    parser.add_argument('image_path', help='Path to the PNG image')
    parser.add_argument('--output', '-o', default='data/output/orientation_test', 
                       help='Output directory for rotated images (default: data/output/orientation_test)')
    
    args = parser.parse_args()
    
    # Check if image file exists
    if not os.path.exists(args.image_path):
        print(f"❌ Image file not found: {args.image_path}")
        sys.exit(1)
    
    try:
        # Test image orientation
        test_image_orientation(args.image_path, args.output)
        
    except Exception as e:
        print(f"❌ Error testing image orientation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
