#!/usr/bin/env python3
"""
Script 1: PDF to PNG Conversion with Interactive Orientation Selection
Converts PDF files to high-quality PNG images and always asks user for rotation preference.
"""

import os
import sys
import logging
from pdf2image import convert_from_path
import argparse
import cv2
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_image_orientation(image):
    """
    Analyze image orientation and suggest rotation.
    
    Args:
        image: PIL Image object
        
    Returns:
        dict: Orientation analysis results
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Get dimensions
    height, width = img_array.shape[:2]
    
    # Calculate aspect ratio
    aspect_ratio = height / width
    
    # Analyze orientation
    if aspect_ratio > 1.5:  # Portrait (tall)
        orientation = "portrait"
        suggestion = "clockwise_90" if width < height else "counter_clockwise_90"
        reason = "Image is very tall (portrait) - consider rotating to landscape for better table processing"
    elif aspect_ratio < 0.7:  # Landscape (wide)
        orientation = "landscape"
        suggestion = "no_rotation"
        reason = "Image is already in landscape orientation - good for table processing"
    else:  # Square-ish
        orientation = "square"
        suggestion = "no_rotation"
        reason = "Image has balanced proportions - no rotation needed"
    
    return {
        "orientation": orientation,
        "dimensions": (width, height),
        "aspect_ratio": aspect_ratio,
        "suggested_rotation": suggestion,
        "reason": reason
    }

def rotate_image(image, rotation_type):
    """
    Rotate image based on rotation type.
    
    Args:
        image: PIL Image object
        rotation_type: Type of rotation to apply
        
    Returns:
        PIL Image: Rotated image
    """
    if rotation_type == "no_rotation":
        return image
    elif rotation_type == "clockwise_90":
        return image.rotate(-90, expand=True)
    elif rotation_type == "counter_clockwise_90":
        return image.rotate(90, expand=True)
    elif rotation_type == "180":
        return image.rotate(180, expand=True)
    else:
        return image

def get_user_rotation_preference(image_analysis):
    """
    Get user input for rotation preference.
    
    Args:
        image_analysis: Dictionary with orientation analysis
        
    Returns:
        str: User's rotation choice
    """
    print(f"\n🔄 Image Orientation Analysis:")
    print(f"   Dimensions: {image_analysis['dimensions'][0]} × {image_analysis['dimensions'][1]}")
    print(f"   Orientation: {image_analysis['orientation'].title()}")
    print(f"   Aspect Ratio: {image_analysis['aspect_ratio']:.2f}")
    print(f"   Suggestion: {image_analysis['reason']}")
    
    print(f"\n📋 Rotation Options:")
    print(f"   1. No rotation (keep original)")
    print(f"   2. Rotate 90° clockwise")
    print(f"   3. Rotate 90° counter-clockwise")
    print(f"   4. Rotate 180°")
    print(f"   5. Use suggested rotation: {image_analysis['suggested_rotation']}")
    
    while True:
        try:
            choice = input(f"\n🎯 Enter your choice (1-5): ").strip()
            
            if choice == "1":
                return "no_rotation"
            elif choice == "2":
                return "clockwise_90"
            elif choice == "3":
                return "counter_clockwise_90"
            elif choice == "4":
                return "180"
            elif choice == "5":
                return image_analysis['suggested_rotation']
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, 4, or 5.")
        except KeyboardInterrupt:
            print("\n\n⚠️ Operation cancelled by user.")
            return "no_rotation"
        except Exception as e:
            print(f"❌ Error: {e}. Please try again.")

def convert_pdf_to_png(pdf_path: str, output_dir: str, dpi: int = 300):
    """
    Convert PDF to PNG images with orientation detection and user choice.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save PNG images
        dpi: Resolution for image conversion (default: 300)
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"🔄 Converting PDF to PNG: {os.path.basename(pdf_path)}")
        logger.info(f"   Output directory: {output_dir}")
        logger.info(f"   DPI: {dpi}")
        
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=dpi)
        
        logger.info(f"   ✅ Converted {len(images)} pages to images")
        
        # Save each page as PNG with orientation handling
        saved_files = []
        rotated_files = []
        
        for i, image in enumerate(images):
            # Analyze orientation
            orientation_analysis = analyze_image_orientation(image)
            
            logger.info(f"\n📐 Page {i+1} Orientation Analysis:")
            logger.info(f"   Dimensions: {orientation_analysis['dimensions'][0]} × {orientation_analysis['dimensions'][1]}")
            logger.info(f"   Orientation: {orientation_analysis['orientation'].title()}")
            logger.info(f"   Suggestion: {orientation_analysis['suggested_rotation']}")
            
            # Always ask user for rotation preference
            rotation_choice = get_user_rotation_preference(orientation_analysis)
            logger.info(f"   🔄 User chose rotation: {rotation_choice}")
            
            # Apply rotation
            rotated_image = rotate_image(image, rotation_choice)
            
            # Save original image
            original_filename = f"page_{i+1:03d}_original.png"
            original_filepath = os.path.join(output_dir, original_filename)
            image.save(original_filepath, 'PNG')
            saved_files.append(original_filepath)
            
            # Save rotated image
            rotated_filename = f"page_{i+1:03d}_rotated.png"
            rotated_filepath = os.path.join(output_dir, rotated_filename)
            rotated_image.save(rotated_filepath, 'PNG')
            rotated_files.append(rotated_filepath)
            
            logger.info(f"      💾 Saved original page {i+1}: {original_filename}")
            logger.info(f"      💾 Saved rotated page {i+1}: {rotated_filename}")
            
            # Save orientation info for this page
            info_filename = f"page_{i+1:03d}_orientation_info.txt"
            info_filepath = os.path.join(output_dir, info_filename)
            with open(info_filepath, 'w') as f:
                f.write(f"Page {i+1} Orientation Analysis:\n")
                f.write(f"Original Dimensions: {orientation_analysis['dimensions'][0]} × {orientation_analysis['dimensions'][1]}\n")
                f.write(f"Orientation: {orientation_analysis['orientation']}\n")
                f.write(f"Aspect Ratio: {orientation_analysis['aspect_ratio']:.2f}\n")
                f.write(f"Suggested Rotation: {orientation_analysis['suggested_rotation']}\n")
                f.write(f"Applied Rotation: {rotation_choice}\n")
                f.write(f"Reason: {orientation_analysis['reason']}\n")
            
            logger.info(f"      📝 Orientation info saved: {info_filename}")
        
        # Save overall conversion info
        info_file = os.path.join(output_dir, "conversion_info.txt")
        with open(info_file, 'w') as f:
            f.write(f"PDF Source: {pdf_path}\n")
            f.write(f"DPI: {dpi}\n")
            f.write(f"Pages: {len(images)}\n")
            f.write(f"Output Directory: {output_dir}\n")
            f.write(f"User Choice: Always interactive rotation selection\n\n")
            f.write(f"Files Generated:\n")
            f.write(f"Original PNG Files:\n")
            for file in saved_files:
                f.write(f"  - {os.path.basename(file)}\n")
            f.write(f"\nRotated PNG Files:\n")
            for file in rotated_files:
                f.write(f"  - {os.path.basename(file)}\n")
        
        logger.info(f"   📝 Conversion info saved to: {info_file}")
        logger.info(f"🎉 PDF to PNG conversion completed successfully!")
        
        return {
            "original_files": saved_files,
            "rotated_files": rotated_files,
            "total_pages": len(images)
        }
        
    except Exception as e:
        logger.error(f"❌ Error converting PDF to PNG: {e}")
        raise

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Convert PDF to PNG images with orientation detection')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--output', '-o', default='data/output/png', 
                       help='Output directory for PNG files (default: data/output/png)')
    parser.add_argument('--dpi', type=int, default=300, 
                       help='DPI for image conversion (default: 300)')
    
    args = parser.parse_args()
    
    # Check if PDF file exists
    if not os.path.exists(args.pdf_path):
        logger.error(f"❌ PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    try:
        # Convert PDF to PNG
        result = convert_pdf_to_png(args.pdf_path, args.output, args.dpi)
        
        print(f"\n🎯 Conversion Summary:")
        print(f"   PDF: {args.pdf_path}")
        print(f"   Total Pages: {result['total_pages']}")
        print(f"   Original PNG Files: {len(result['original_files'])}")
        print(f"   Rotated PNG Files: {len(result['rotated_files'])}")
        print(f"   Output Directory: {args.output}")
        print(f"   DPI: {args.dpi}")
        
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
