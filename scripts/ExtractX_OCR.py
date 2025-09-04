#!/usr/bin/env python3
"""
Complete OCR Pipeline - Interactive Version
Processes PDF/images through the entire pipeline: PDF→PNG→Gemini→Excel/CSV
with user interaction for all options
"""

import os
import sys
import json
import logging
import argparse
import base64
import requests
import io
import time
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompletePipeline:
    def __init__(self, api_key: str = None):
        """Initialize the complete pipeline processor
        
        Args:
            api_key: Gemini API key (optional if GEMINI_API_KEY in environment)
        """
        load_dotenv()
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key required (provide api_key parameter or set GEMINI_API_KEY environment variable)")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        self.headers = {"Content-Type": "application/json"}
        
        # Output directory for temporary files
        self.temp_dir = "data/output/temp"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        logger.info("✅ Complete Pipeline initialized successfully")
    
    def get_user_input_path(self):
        """Interactive function to get input file path from user"""
        print("📁 Select input file:")
        print("1. Enter file path manually")
        print("2. Browse current directory")
        print("3. Browse data/input directory")
        
        while True:
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == "1":
                file_path = input("Enter full path to your PDF/image file: ").strip()
                if os.path.exists(file_path) and file_path.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
                    return os.path.abspath(file_path)
                else:
                    print("❌ File not found or unsupported format. Please enter a valid PDF/PNG/JPG file.")
                    
            elif choice == "2":
                current_dir = "."
                print(f"\nFiles in current directory:")
                files = [f for f in os.listdir(current_dir) if f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg'))]
                if files:
                    for i, f in enumerate(files, 1):
                        print(f"  {i}. {f}")
                    try:
                        idx = int(input("Select file number: ")) - 1
                        if 0 <= idx < len(files):
                            return os.path.abspath(files[idx])
                        else:
                            print("❌ Invalid selection")
                    except ValueError:
                        print("❌ Please enter a valid number")
                else:
                    print("❌ No PDF/image files found in current directory")
                    
            elif choice == "3":
                input_dir = "data/input/"
                if os.path.exists(input_dir):
                    print(f"\nFiles in {input_dir}:")
                    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg'))]
                    if files:
                        for i, f in enumerate(files, 1):
                            print(f"  {i}. {f}")
                        try:
                            idx = int(input("Select file number: ")) - 1
                            if 0 <= idx < len(files):
                                return os.path.join(input_dir, files[idx])
                            else:
                                print("❌ Invalid selection")
                        except ValueError:
                            print("❌ Please enter a valid number")
                    else:
                        print(f"❌ No PDF/image files found in {input_dir}")
                else:
                    print(f"❌ Directory {input_dir} not found")
            else:
                print("❌ Invalid option. Please select 1, 2, or 3")

    def get_page_count(self, file_path):
        """Get the number of pages in a PDF file"""
        if file_path.lower().endswith('.pdf'):
            try:
                doc = fitz.open(file_path)
                page_count = len(doc)
                doc.close()
                print(f"📄 PDF contains {page_count} pages")
                return page_count
            except Exception as e:
                print(f"❌ Error reading PDF: {e}")
                return None
        return 1  # For images

    def get_user_page_selection(self, total_pages):
        """Get user's page selection for processing"""
        print(f"\n📄 Page Selection (Total: {total_pages} pages):")
        print("1. Process all pages")
        print("2. Process specific page range (e.g., 1-5)")
        print("3. Process first N pages")
        
        while True:
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == "1":
                return 1, None  # Process all pages
                
            elif choice == "2":
                range_input = input("Enter page range (e.g., 1-5, 2-8): ").strip()
                try:
                    if '-' in range_input:
                        start, end = map(int, range_input.split('-'))
                        if 1 <= start <= end <= total_pages:
                            return start, end
                        else:
                            print(f"❌ Invalid range. Pages must be between 1 and {total_pages}")
                    else:
                        page_num = int(range_input)
                        if 1 <= page_num <= total_pages:
                            return page_num, page_num
                        else:
                            print(f"❌ Invalid page. Must be between 1 and {total_pages}")
                except ValueError:
                    print("❌ Invalid format. Use format like '1-5' or '3'")
                    
            elif choice == "3":
                try:
                    n_pages = int(input(f"Enter number of pages to process (1-{total_pages}): "))
                    if 1 <= n_pages <= total_pages:
                        return 1, n_pages
                    else:
                        print(f"❌ Invalid number. Must be between 1 and {total_pages}")
                except ValueError:
                    print("❌ Please enter a valid number")
                    
            else:
                print("❌ Invalid option. Please select 1, 2, or 3")

    def get_user_rotation_choice(self):
        """Get user's rotation preference"""
        print("\n🔄 Rotation Options:")
        print("1. No rotation (0°)")
        print("2. Rotate 90° clockwise")
        print("3. Rotate 180°")
        print("4. Rotate 270° clockwise (90° counter-clockwise)")
        
        rotation_map = {"1": 0, "2": 90, "3": 180, "4": 270}
        
        while True:
            choice = input("\nSelect rotation (1-4): ").strip()
            if choice in rotation_map:
                rotation = rotation_map[choice]
                print(f"✅ Selected: {rotation}° rotation")
                return rotation
            else:
                print("❌ Invalid option. Please select 1, 2, 3, or 4")

    def get_user_output_format(self):
        """Get user's output format preference"""
        print("\n💾 Output Format Options:")
        print("1. Excel (.xlsx)")
        print("2. CSV (.csv)")
        print("3. Both Excel and CSV")
        
        while True:
            choice = input("\nSelect format (1-3): ").strip()
            
            if choice == "1":
                return ["excel"]
            elif choice == "2":
                return ["csv"]
            elif choice == "3":
                return ["excel", "csv"]
            else:
                print("❌ Invalid option. Please select 1, 2, or 3")

    def get_user_output_path(self):
        """Get user's output directory preference"""
        print("\n📂 Output Directory:")
        print("1. Use default (data/output/interactive_extraction)")
        print("2. Enter custom path")
        
        while True:
            choice = input("\nSelect option (1-2): ").strip()
            
            if choice == "1":
                default_path = "data/output/interactive_extraction"
                os.makedirs(default_path, exist_ok=True)
                return default_path
                
            elif choice == "2":
                custom_path = input("Enter output directory path: ").strip()
                try:
                    os.makedirs(custom_path, exist_ok=True)
                    return custom_path
                except Exception as e:
                    print(f"❌ Error creating directory: {e}")
                    print("Please try again")
                    
            else:
                print("❌ Invalid option. Please select 1 or 2")
    
    def pdf_to_images(self, pdf_path: str, output_dir: str, rotation: int = 0, dpi: int = 300, start_page: int = 1, end_page: int = None) -> List[str]:
        """Convert PDF to PNG images with page range support"""
        logger.info(f"🔄 Converting PDF: {os.path.basename(pdf_path)}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open PDF
        doc = fitz.open(pdf_path)
        image_paths = []
        
        # Set page range
        total_pages = len(doc)
        if end_page is None:
            end_page = total_pages
        
        logger.info(f"   Processing pages {start_page} to {end_page} of {total_pages}")
        
        for page_num in range(start_page - 1, min(end_page, total_pages)):
            try:
                page = doc[page_num]
                
                # Convert to image
                mat = fitz.Matrix(dpi/72, dpi/72)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Create PIL Image for rotation
                img = Image.open(io.BytesIO(img_data))
                
                # Apply rotation if specified
                if rotation != 0:
                    img = img.rotate(-rotation, expand=True)
                
                # Save image
                page_filename = f"page_{page_num + 1:03d}.png"
                image_path = os.path.join(output_dir, page_filename)
                img.save(image_path)
                image_paths.append(image_path)
                
                logger.info(f"   ✅ Page {page_num + 1} → {page_filename}")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to process page {page_num + 1}: {e}")
                continue
        
        doc.close()
        logger.info(f"📄 PDF conversion complete: {len(image_paths)} images created")
        return image_paths
    
    def process_single_image(self, image_path: str, rotation: int = 0) -> str:
        """Process a single image file with rotation"""
        logger.info(f"🖼️ Processing image: {os.path.basename(image_path)}")
        
        if rotation == 0:
            return image_path
        
        try:
            # Load and rotate image
            img = Image.open(image_path)
            rotated_img = img.rotate(-rotation, expand=True)
            
            # Save rotated image
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            rotated_path = os.path.join(self.temp_dir, f"{base_name}_rotated_{rotation}.png")
            rotated_img.save(rotated_path)
            
            logger.info(f"   ✅ Rotated {rotation}° → {os.path.basename(rotated_path)}")
            return rotated_path
            
        except Exception as e:
            logger.error(f"   ❌ Failed to rotate image: {e}")
            return image_path
    
    def extract_table_with_gemini(self, image_path: str) -> dict:
        """Extract table data using Gemini Vision API with retry logic"""
        logger.info(f"🤖 Extracting table with Gemini: {os.path.basename(image_path)}")
        
        # Encode image to base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        prompt = """
        You are an expert table extraction system. Analyze this image and extract ALL tabular data you find.

        CRITICAL REQUIREMENTS:
        1. Extract EVERY row and column, including headers
        2. Maintain exact data relationships and positioning
        3. Handle merged cells by repeating values appropriately
        4. Preserve numerical values, dates, and text exactly as shown
        5. If there are multiple tables, combine them logically

        Return the data as a JSON object with this exact structure:
        {
            "table_data": [
                {"column1": "value1", "column2": "value2", ...},
                {"column1": "value3", "column2": "value4", ...}
            ],
            "summary": {
                "total_rows": number,
                "total_columns": number,
                "confidence": "high/medium/low"
            }
        }

        IMPORTANT: Return ONLY the JSON object, no additional text or formatting.
        """
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_data
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 8192
            }
        }
        
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.gemini_url}?key={self.api_key}",
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'candidates' in result and len(result['candidates']) > 0:
                        text_response = result['candidates'][0]['content']['parts'][0]['text']
                        
                        # Extract JSON from response
                        try:
                            # Try to find JSON in the response
                            json_start = text_response.find('{')
                            json_end = text_response.rfind('}') + 1
                            
                            if json_start >= 0 and json_end > json_start:
                                json_str = text_response[json_start:json_end]
                                gemini_data = json.loads(json_str)
                                
                                logger.info(f"   ✅ Extraction successful")
                                return gemini_data
                            else:
                                logger.warning(f"   ⚠️ No valid JSON found in response")
                                return {"table_data": [], "summary": {"total_rows": 0, "total_columns": 0, "confidence": "low"}}
                                
                        except json.JSONDecodeError as e:
                            logger.warning(f"   ⚠️ JSON parsing failed: {e}")
                            return {"table_data": [], "summary": {"total_rows": 0, "total_columns": 0, "confidence": "low"}}
                    else:
                        logger.warning(f"   ⚠️ No candidates in Gemini response")
                        return {"table_data": [], "summary": {"total_rows": 0, "total_columns": 0, "confidence": "low"}}
                
                elif response.status_code in [503, 429, 500, 502, 504]:
                    # Server errors - retry with exponential backoff
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"   ⚠️ Server error {response.status_code}, retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"   ❌ Max retries reached. Status: {response.status_code}")
                        return {"table_data": [], "summary": {"total_rows": 0, "total_columns": 0, "confidence": "low"}}
                else:
                    logger.error(f"   ❌ API request failed with status {response.status_code}: {response.text}")
                    return {"table_data": [], "summary": {"total_rows": 0, "total_columns": 0, "confidence": "low"}}
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"   ⚠️ Request failed: {e}, retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"   ❌ Request failed after {max_retries} attempts: {e}")
                    return {"table_data": [], "summary": {"total_rows": 0, "total_columns": 0, "confidence": "low"}}
        
        return {"table_data": [], "summary": {"total_rows": 0, "total_columns": 0, "confidence": "low"}}
    
    def create_dataframe_from_gemini(self, gemini_data: dict) -> pd.DataFrame:
        """Convert Gemini extracted data to pandas DataFrame"""
        try:
            if not gemini_data or 'table_data' not in gemini_data or not gemini_data['table_data']:
                logger.warning("   ⚠️ No table data found in Gemini response")
                return pd.DataFrame()
            
            df = pd.DataFrame(gemini_data['table_data'])
            
            if df.empty:
                logger.warning("   ⚠️ DataFrame is empty")
                return df
            
            # Clean the data
            df = df.replace(['', 'nan', 'NaN', 'null', 'NULL'], pd.NA)
            
            logger.info(f"   📊 Created DataFrame: {df.shape[0]} rows × {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"   ❌ Failed to create DataFrame: {e}")
            return pd.DataFrame()
    
    def save_to_excel(self, df: pd.DataFrame, file_path: str):
        """Save DataFrame to Excel with formatting"""
        try:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Extracted_Data', index=False)
                
                # Get the worksheet to apply formatting
                worksheet = writer.sheets['Extracted_Data']
                
                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    
                    # Set width with some padding, max 50 characters
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
            
            logger.info(f"💾 Excel saved: {os.path.basename(file_path)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save Excel: {e}")
    
    def save_to_csv(self, df: pd.DataFrame, file_path: str):
        """Save DataFrame to CSV"""
        try:
            df.to_csv(file_path, index=False, encoding='utf-8')
            logger.info(f"💾 CSV saved: {os.path.basename(file_path)}")
        except Exception as e:
            logger.error(f"❌ Failed to save CSV: {e}")
    
    def save_outputs(self, df: pd.DataFrame, output_dir: str, base_name: str, formats: List[str]):
        """Save DataFrame in requested formats with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for format_type in formats:
            if format_type == "excel":
                file_path = os.path.join(output_dir, f"{base_name}_{timestamp}.xlsx")
                self.save_to_excel(df, file_path)
            elif format_type == "csv":
                file_path = os.path.join(output_dir, f"{base_name}_{timestamp}.csv")
                self.save_to_csv(df, file_path)
    
    def process_files(self, input_path: str, output_dir: str, rotation: int = 0, dpi: int = 300, start_page: int = 1, end_page: int = None, output_formats: List[str] = ["excel"]) -> pd.DataFrame:
        """Process file with user-specified options and combine results"""
        os.makedirs(output_dir, exist_ok=True)
        
        all_dataframes = []
        processed_images = []
        
        logger.info(f"📁 Processing: {input_path}")
        
        if input_path.lower().endswith('.pdf'):
            # PDF processing with page range
            pdf_output_dir = os.path.join(output_dir, f"{Path(input_path).stem}_images")
            image_paths = self.pdf_to_images(input_path, pdf_output_dir, rotation, dpi, start_page, end_page)
            processed_images.extend(image_paths)
        
        elif input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Image processing
            processed_image = self.process_single_image(input_path, rotation)
            processed_images.append(processed_image)
        
        else:
            logger.error(f"❌ Unsupported file type: {input_path}")
            return pd.DataFrame()
        
        # Extract tables from all images
        logger.info(f"🔍 Extracting tables from {len(processed_images)} images...")
        
        for i, image_path in enumerate(processed_images, 1):
            logger.info(f"[{i}/{len(processed_images)}] Processing: {os.path.basename(image_path)}")
            
            gemini_data = self.extract_table_with_gemini(image_path)
            df = self.create_dataframe_from_gemini(gemini_data)
            
            if not df.empty:
                logger.info(f"   ✅ Extracted: {df.shape[0]} rows × {df.shape[1]} columns")
                all_dataframes.append(df)
            else:
                logger.info(f"   ⚠️ No table data found")
        
        if not all_dataframes:
            logger.warning("❌ No tables extracted from any images")
            return pd.DataFrame()
        
        # Combine all DataFrames
        logger.info(f"📊 Combining {len(all_dataframes)} tables...")
        
        if len(all_dataframes) == 1:
            combined_df = all_dataframes[0]
        else:
            # Align columns across all DataFrames
            all_columns = set()
            for df in all_dataframes:
                all_columns.update(df.columns)
            
            # Ensure all DataFrames have the same columns
            for df in all_dataframes:
                for col in all_columns:
                    if col not in df.columns:
                        df[col] = ""
            
            # Reorder columns consistently
            column_order = list(all_columns)
            aligned_dataframes = [df[column_order] for df in all_dataframes]
            
            # Combine DataFrames
            combined_df = pd.concat(aligned_dataframes, ignore_index=True)
        
        logger.info(f"🎉 Combined result: {combined_df.shape[0]} rows × {combined_df.shape[1]} columns")
        
        # Save in requested formats
        self.save_outputs(combined_df, output_dir, "interactive_extraction", output_formats)
        
        return combined_df

def main():
    """Interactive OCR Pipeline - User guided extraction with all options"""
    print("=" * 80)
    print("🔍 ExtractX - Interactive OCR Pipeline")
    print("📄 Supports: PDF, PNG, JPG, JPEG | 🤖 Powered by Gemini AI")
    print("=" * 80)
    print()
    
    # Initialize pipeline
    try:
        processor = CompletePipeline()
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        print("💡 Make sure you have a valid GEMINI_API_KEY in your .env file or environment")
        return
    
    # Step 1: Get input file path
    input_path = processor.get_user_input_path()
    if not input_path:
        print("❌ No valid input file selected. Exiting.")
        return
    
    # Step 2: Check pages (for PDF) and get user selection
    start_page, end_page = 1, None
    
    if input_path.lower().endswith('.pdf'):
        total_pages = processor.get_page_count(input_path)
        if total_pages:
            start_page, end_page = processor.get_user_page_selection(total_pages)
        else:
            print("❌ Could not read PDF file. Exiting.")
            return
    
    # Step 3: Get rotation choice
    rotation = processor.get_user_rotation_choice()
    
    # Step 4: Get output format
    output_formats = processor.get_user_output_format()
    
    # Step 5: Get output directory
    output_dir = processor.get_user_output_path()
    
    # Display processing summary
    print("\n" + "=" * 50)
    print("📋 PROCESSING SUMMARY")
    print("=" * 50)
    print(f"📁 Input File: {input_path}")
    if input_path.lower().endswith('.pdf'):
        if end_page:
            print(f"📄 Pages: {start_page} to {end_page}")
        else:
            print(f"📄 Pages: {start_page} onwards")
    print(f"🔄 Rotation: {rotation}°")
    print(f"💾 Output Format(s): {', '.join(output_formats).upper()}")
    print(f"📂 Output Directory: {output_dir}")
    print("=" * 50)
    
    proceed = input("\n▶️ Proceed with extraction? (y/n): ").strip().lower()
    if proceed not in ['y', 'yes']:
        print("❌ Extraction cancelled by user.")
        return
    
    # Process the file
    print("\n🚀 Starting extraction process...")
    print("-" * 50)
    
    try:
        result_df = processor.process_files(
            input_path=input_path,
            output_dir=output_dir,
            rotation=rotation,
            start_page=start_page,
            end_page=end_page,
            output_formats=output_formats
        )
        
        if not result_df.empty:
            print("\n" + "=" * 50)
            print("🎉 EXTRACTION COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            print(f"📊 Total Records Extracted: {result_df.shape[0]}")
            print(f"📈 Total Columns: {result_df.shape[1]}")
            
            if len(result_df.columns) > 0:
                print(f"📋 Columns: {', '.join(result_df.columns[:5])}")
                if len(result_df.columns) > 5:
                    print(f"          ... and {len(result_df.columns) - 5} more")
            
            print(f"💾 Output saved in: {output_dir}")
            print("=" * 50)
        else:
            print("\n❌ No data extracted. Please check your input file and try again.")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        logger.error(f"Processing error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
