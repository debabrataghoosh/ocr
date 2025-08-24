#!/usr/bin/env python3
"""
Master Script: Complete Table Extraction Pipeline
Runs all steps in sequence: PDF→PNG→Grid Detection→Text Extraction→Excel Export
"""

import os
import sys
import logging
import argparse
import subprocess
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_script(script_path: str, args: list, description: str):
    """
    Run a Python script with given arguments.
    
    Args:
        script_path: Path to the script to run
        args: List of command line arguments
        description: Description of what the script does
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"🚀 Running: {description}")
        logger.info(f"   Script: {os.path.basename(script_path)}")
        logger.info(f"   Arguments: {' '.join(args)}")
        
        # Run the script
        result = subprocess.run([sys.executable, script_path] + args, 
                              capture_output=True, text=True, check=True)
        
        # Log output
        if result.stdout:
            logger.info(f"   ✅ Output: {result.stdout.strip()}")
        
        logger.info(f"   ✅ {description} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"   ❌ {description} failed with exit code {e.returncode}")
        if e.stdout:
            logger.error(f"   STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"   STDERR: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"   ❌ Error running {description}: {e}")
        return False

def run_complete_pipeline(pdf_path: str, output_base_dir: str = "data/output"):
    """
    Run the complete table extraction pipeline.
    
    Args:
        pdf_path: Path to the input PDF file
        output_base_dir: Base directory for all outputs
    """
    try:
        logger.info("🎯 Starting Complete Table Extraction Pipeline")
        logger.info("=" * 70)
        logger.info(f"📄 Input PDF: {pdf_path}")
        logger.info(f"📁 Output Base Directory: {output_base_dir}")
        logger.info("=" * 70)
        
        # Define output directories for each step
        png_dir = os.path.join(output_base_dir, "png")
        grid_dir = os.path.join(output_base_dir, "grid_detection")
        text_dir = os.path.join(output_base_dir, "text_extraction")
        excel_dir = os.path.join(output_base_dir, "excel")
        
        # Step 1: PDF to PNG Conversion
        logger.info("\n📋 Step 1: PDF to PNG Conversion")
        logger.info("-" * 50)
        
        if not run_script("scripts/01_pdf_to_png.py", 
                         [pdf_path, "--output", png_dir], 
                         "PDF to PNG conversion"):
            logger.error("❌ Pipeline failed at Step 1")
            return False
        
        # Find the generated PNG file
        png_files = [f for f in os.listdir(png_dir) if f.endswith('.png') and f.startswith('page_')]
        if not png_files:
            logger.error("❌ No PNG files generated")
            return False
        
        png_file = os.path.join(png_dir, png_files[0])  # Use first page
        logger.info(f"   📸 PNG file: {png_file}")
        
        # Step 2: Table Grid Detection
        logger.info("\n📋 Step 2: Table Grid Detection")
        logger.info("-" * 50)
        
        if not run_script("scripts/02_detect_table_grid.py", 
                         [png_file, "--output", grid_dir], 
                         "Table grid detection"):
            logger.error("❌ Pipeline failed at Step 2")
            return False
        
        # Find the generated grid file
        grid_files = [f for f in os.listdir(grid_dir) if f.startswith('04_combined_grid')]
        if not grid_files:
            logger.error("❌ No grid files generated")
            return False
        
        grid_file = os.path.join(grid_dir, grid_files[0])
        logger.info(f"   🔍 Grid file: {grid_file}")
        
        # Step 3: Cell Identification and Text Extraction
        logger.info("\n📋 Step 3: Cell Identification and Text Extraction")
        logger.info("-" * 50)
        
        if not run_script("scripts/03_extract_text_from_cells.py", 
                         [grid_file, png_file, "--output", text_dir], 
                         "Text extraction from cells"):
            logger.error("❌ Pipeline failed at Step 3")
            return False
        
        # Find the generated text data file
        text_files = [f for f in os.listdir(text_dir) if f == "extracted_text.json"]
        if not text_files:
            logger.error("❌ No text data files generated")
            return False
        
        text_file = os.path.join(text_dir, text_files[0])
        logger.info(f"   🔤 Text data file: {text_file}")
        
        # Step 4: Table Reconstruction and Excel Export
        logger.info("\n📋 Step 4: Table Reconstruction and Excel Export")
        logger.info("-" * 50)
        
        if not run_script("scripts/04_create_excel_table.py", 
                         [text_file, "--output", excel_dir], 
                         "Excel table creation"):
            logger.error("❌ Pipeline failed at Step 4")
            return False
        
        # Find the generated Excel file
        excel_files = [f for f in os.listdir(excel_dir) if f.endswith('.xlsx')]
        if not excel_files:
            logger.error("❌ No Excel files generated")
            return False
        
        excel_file = os.path.join(excel_dir, excel_files[0])
        logger.info(f"   💾 Excel file: {excel_file}")
        
        # Pipeline completed successfully
        logger.info("\n🎉 Complete Pipeline Summary")
        logger.info("=" * 70)
        logger.info(f"✅ PDF to PNG: {png_file}")
        logger.info(f"✅ Grid Detection: {grid_file}")
        logger.info(f"✅ Text Extraction: {text_file}")
        logger.info(f"✅ Excel Export: {excel_file}")
        logger.info(f"📁 All outputs saved to: {output_base_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {e}")
        return False

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Run complete table extraction pipeline')
    parser.add_argument('pdf_path', help='Path to the input PDF file')
    parser.add_argument('--output', '-o', default='data/output', 
                       help='Base output directory (default: data/output)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if PDF file exists
    if not os.path.exists(args.pdf_path):
        logger.error(f"❌ PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    # Check if scripts directory exists
    if not os.path.exists("scripts"):
        logger.error("❌ Scripts directory not found. Please run from project root.")
        sys.exit(1)
    
    # Check if all required scripts exist
    required_scripts = [
        "scripts/01_pdf_to_png.py",
        "scripts/02_detect_table_grid.py", 
        "scripts/03_extract_text_from_cells.py",
        "scripts/04_create_excel_table.py"
    ]
    
    missing_scripts = [script for script in required_scripts if not os.path.exists(script)]
    if missing_scripts:
        logger.error(f"❌ Missing required scripts: {', '.join(missing_scripts)}")
        sys.exit(1)
    
    # Start timing
    start_time = time.time()
    
    try:
        # Run the complete pipeline
        success = run_complete_pipeline(args.pdf_path, args.output)
        
        if success:
            # Calculate execution time
            execution_time = time.time() - start_time
            logger.info(f"\n⏱️ Total execution time: {execution_time:.2f} seconds")
            logger.info("🎯 Pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Pipeline failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
