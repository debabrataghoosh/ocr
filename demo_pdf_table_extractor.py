#!/usr/bin/env python3
"""
Demo script for PDF Table Extractor with Advanced Image Processing
Uses OpenCV for grid detection and Tesseract for cell-by-cell OCR
"""

import os
import sys
import pandas as pd
from src.pdf_table_extractor import PDFTableExtractor

def demo_pdf_table_extraction():
    """Demonstrate PDF table extraction using advanced image processing."""
    print("🎯 PDF Table Extractor Demo - Advanced Image Processing")
    print("=" * 70)
    
    # Check if PDF file exists
    pdf_path = "data/input/daat12221.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        print("💡 Please ensure the PDF file is in the data/input/ directory")
        return False
    
    try:
        # Initialize PDF table extractor
        print("🔧 Initializing PDF Table Extractor...")
        extractor = PDFTableExtractor()
        
        # Show what we're going to do
        print(f"\n📋 PDF Table Extraction Plan:")
        print(f"   1. Convert PDF to high-quality images")
        print(f"   2. Prepare images for line detection (grayscale, threshold, invert)")
        print(f"   3. Detect horizontal and vertical grid lines using OpenCV")
        print(f"   4. Identify every cell in the grid using contours")
        print(f"   5. Perform OCR on each cell individually with Tesseract")
        print(f"   6. Rebuild table structure and export to Excel")
        
        # Extract all tables
        print(f"\n🚀 Starting advanced table extraction...")
        tables = extractor.extract_tables_from_pdf(pdf_path)
        
        if not tables:
            print("❌ No tables could be extracted from the PDF")
            return False
        
        # Show extraction summary
        print(f"\n📊 Table Extraction Summary:")
        print("-" * 50)
        summary = extractor.get_table_summary()
        print(f"Total tables extracted: {summary['total_tables']}")
        
        for table_info in summary['tables']:
            print(f"\nTable {table_info['table_number']}:")
            print(f"  Dimensions: {table_info['dimensions']}")
            print(f"  Total cells: {table_info['total_cells']}")
            print(f"  Non-empty cells: {table_info['non_empty_cells']}")
            
            # Show sample data
            if table_info['sample_data']:
                print(f"  Sample data (first 3 rows):")
                for i, row in enumerate(table_info['sample_data'][:3]):
                    print(f"    Row {i+1}: {row}")
        
        # Export to Excel with formatting
        print(f"\n💾 Exporting tables to Excel...")
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        excel_filename = f"advanced_extraction_tables_{timestamp}.xlsx"
        excel_path = os.path.join("data/output/exel", excel_filename)
        
        excel_path = extractor.export_to_excel(excel_path)
        
        print(f"\n🎉 PDF table extraction completed successfully!")
        print(f"📁 Excel file: {excel_path}")
        print(f"📊 Tables extracted: {len(tables)}")
        
        return True
        
    except Exception as e:
        print(f"❌ PDF table extraction demo failed: {e}")
        return False

def demo_extraction_approach():
    """Demonstrate the new extraction approach."""
    print("\n🔍 Advanced Table Extraction Approach")
    print("=" * 70)
    
    print("📋 Step-by-Step Process:")
    print("   1. Image Preparation for Line Detection:")
    print("      → Load image and correct rotation")
    print("      → Convert to grayscale")
    print("      → Apply adaptive thresholding")
    print("      → Invert image (lines white, background black)")
    print("      → Thicken lines using dilation")
    print()
    print("   2. Grid Line Detection:")
    print("      → Isolate horizontal lines with morphological operations")
    print("      → Isolate vertical lines with morphological operations")
    print("      → Combine to form complete grid")
    print()
    print("   3. Cell Identification:")
    print("      → Find cell contours using OpenCV")
    print("      → Sort contours top-to-bottom, left-to-right")
    print("      → Get precise bounding boxes for each cell")
    print()
    print("   4. Individual Cell OCR:")
    print("      → Crop each cell from original image")
    print("      → Run Tesseract on individual cells")
    print("      → Use PSM 7 (single line) for accuracy")
    print()
    print("   5. Table Reconstruction:")
    print("      → Organize cell text by position")
    print("      → Create structured DataFrame")
    print("      → Export to Excel with proper formatting")
    print()
    print("💡 Key Advantages:")
    print("   - Cell-by-cell OCR for maximum accuracy")
    print("   - Precise grid detection using OpenCV")
    print("   - Maintains table structure and alignment")
    print("   - Works with scanned documents and complex tables")

def main():
    """Main demo function."""
    print("🚀 PDF Table Extractor with Advanced Image Processing")
    print("=" * 70)
    
    # Check if required libraries are available
    try:
        import cv2
        print("✅ OpenCV is available")
    except ImportError:
        print("❌ OpenCV not available")
        return
    
    try:
        import pytesseract
        print("✅ Tesseract OCR is available")
    except ImportError:
        print("❌ Tesseract OCR not available")
        return
    
    try:
        from pdf2image import convert_from_path
        print("✅ pdf2image is available")
    except ImportError:
        print("❌ pdf2image not available")
        return
    
    # Run PDF table extraction demo
    success = demo_pdf_table_extraction()
    
    if success:
        demo_extraction_approach()
        
        print("\n🎯 Advanced Table Extraction Demo Summary")
        print("=" * 70)
        print("✅ PDF table extraction completed successfully")
        print("📊 Tables extracted using advanced image processing")
        print("🔍 Grid detection with OpenCV for precise cell identification")
        print("🔤 Cell-by-cell OCR with Tesseract for maximum accuracy")
        print("📁 Excel file maintains table structure and alignment")
        print("\n💡 Key Benefits:")
        print("   - Works with scanned documents and complex tables")
        print("   - Precise grid line detection using computer vision")
        print("   - Individual cell OCR for better text accuracy")
        print("   - Maintains table structure and cell alignment")
        print("\n🚀 This approach gives you Excel that accurately represents your PDF tables!")
        print("   Perfect for scanned documents, complex layouts, and precise extraction.")
    else:
        print("\n❌ PDF table extraction demo failed")
        print("💡 Check the error messages above")

if __name__ == "__main__":
    main()
