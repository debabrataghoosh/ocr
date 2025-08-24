#!/usr/bin/env python3
"""
Script 1: PDF to PNG Conversion
Converts PDF files to high-quality PNG images for processing.
"""

import os
import sys
import logging
from pdf2image import convert_from_path
import argparse

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_pdf_to_png(pdf_path: str, output_dir: str, dpi: int = 300):
    """
    Convert PDF to PNG images.
    
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
        
        # Save each page as PNG
        saved_files = []
        for i, image in enumerate(images):
            filename = f"page_{i+1:03d}.png"
            filepath = os.path.join(output_dir, filename)
            
            image.save(filepath, 'PNG')
            saved_files.append(filepath)
            
            logger.info(f"      💾 Saved page {i+1}: {filename}")
        
        # Save conversion info
        info_file = os.path.join(output_dir, "conversion_info.txt")
        with open(info_file, 'w') as f:
            f.write(f"PDF Source: {pdf_path}\n")
            f.write(f"DPI: {dpi}\n")
            f.write(f"Pages: {len(images)}\n")
            f.write(f"Output Directory: {output_dir}\n")
            f.write(f"PNG Files:\n")
            for file in saved_files:
                f.write(f"  - {os.path.basename(file)}\n")
        
        logger.info(f"   📝 Conversion info saved to: {info_file}")
        logger.info(f"🎉 PDF to PNG conversion completed successfully!")
        
        return saved_files
        
    except Exception as e:
        logger.error(f"❌ Error converting PDF to PNG: {e}")
        raise

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Convert PDF to PNG images')
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
        saved_files = convert_pdf_to_png(args.pdf_path, args.output, args.dpi)
        
        print(f"\n🎯 Conversion Summary:")
        print(f"   PDF: {args.pdf_path}")
        print(f"   PNG Files: {len(saved_files)}")
        print(f"   Output Directory: {args.output}")
        print(f"   DPI: {args.dpi}")
        
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
