#!/usr/bin/env python3
"""
Script 3 (Gemini): Gemini API-Powered Table Extraction
Uses Google Gemini API for accurate table structure detection and text extraction.
This approach is superior to traditional OCR for complex table layouts.
"""

import os
import sys
import json
import logging
import argparse
import base64
import requests
from PIL import Image
import io
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiTableExtractor:
    def __init__(self, api_key: str):
        """
        Initialize Gemini API client.
        """
        self.api_key = api_key
        # Updated endpoint for the correct Gemini model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        self.headers = {
            "Content-Type": "application/json"
        }
        
    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 for Gemini API.
        """
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            logger.error(f"❌ Error encoding image: {e}")
            raise
    
    def extract_table_with_gemini(self, image_path: str, output_dir: str):
        """
        Extract table data using Gemini API.
        """
        try:
            logger.info(f"🔍 Gemini API table extraction from: {os.path.basename(image_path)}")
            
            # Encode image
            logger.info("   🔧 Encoding image for Gemini API...")
            base64_image = self.encode_image(image_path)
            
            # Create Gemini API request with improved prompt
            prompt = """
            You are an expert at analyzing table images and extracting structured data. 

            Analyze this image and extract the table data in the following JSON format:

            {
                "table_info": {
                    "total_rows": <number>,
                    "total_columns": <number>,
                    "headers": ["header1", "header2", "header3", ...]
                },
                "data": [
                    {
                        "row": 1,
                        "cells": ["cell1_value", "cell2_value", "cell3_value", ...]
                    },
                    {
                        "row": 2,
                        "cells": ["cell1_value", "cell2_value", "cell3_value", ...]
                    }
                ]
            }

            IMPORTANT INSTRUCTIONS:
            1. Look carefully at the image and identify ALL table content
            2. Extract column headers from the first row
            3. Extract ALL data from each row
            4. Be very accurate with text extraction
            5. Return ONLY the JSON, no other text
            6. If you see handwritten text, transcribe it accurately
            7. Include all visible information in the table

            Return the data in the exact JSON format shown above.
            """
            
            # Prepare request payload
            payload = {
                "contents": [{
                    "parts": [
                        {
                            "text": prompt
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": base64_image
                            }
                        }
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "topK": 1,
                    "topP": 1,
                    "maxOutputTokens": 8192
                }
            }
            
            # Make API request
            logger.info("   🚀 Sending request to Gemini API...")
            response = requests.post(
                f"{self.base_url}?key={self.api_key}",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"❌ Gemini API request failed: {response.status_code}")
                logger.error(f"   Response: {response.text}")
                raise Exception(f"Gemini API request failed: {response.status_code}")
            
            # Parse response
            logger.info("   ✅ Gemini API response received")
            response_data = response.json()
            
            # Extract text content
            if 'candidates' in response_data and len(response_data['candidates']) > 0:
                content = response_data['candidates'][0]['content']
                if 'parts' in content and len(content['parts']) > 0:
                    extracted_text = content['parts'][0]['text']
                    
                    # Try to parse JSON from response
                    table_data = self.parse_gemini_response(extracted_text)
                    
                    # Save results
                    self.save_gemini_results(table_data, extracted_text, output_dir)
                    
                    return table_data
                else:
                    raise Exception("No content parts in Gemini response")
            else:
                raise Exception("No candidates in Gemini response")
                
        except Exception as e:
            logger.error(f"❌ Gemini API extraction failed: {e}")
            raise
    
    def parse_gemini_response(self, response_text: str) -> dict:
        """
        Parse Gemini API response to extract table data.
        """
        try:
            logger.info("   🔍 Parsing Gemini API response...")
            
            # Look for JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                table_data = json.loads(json_str)
                logger.info("   ✅ Successfully parsed JSON response")
                return table_data
            else:
                # If no JSON found, create structured data from text
                logger.warning("   ⚠️ No JSON found in response, creating structured data...")
                return self.create_structured_data_from_text(response_text)
                
        except json.JSONDecodeError as e:
            logger.warning(f"   ⚠️ JSON parsing failed: {e}")
            logger.info("   🔄 Creating structured data from text response...")
            return self.create_structured_data_from_text(response_text)
    
    def create_structured_data_from_text(self, text: str) -> dict:
        """
        Create structured table data from text response.
        """
        try:
            # Split text into lines
            lines = text.strip().split('\n')
            
            # Find table-like structure
            table_lines = []
            for line in lines:
                if '|' in line or '\t' in line or len(line.split()) > 2:
                    table_lines.append(line)
            
            if not table_lines:
                # If no clear table structure, treat as single column
                table_lines = [line for line in lines if line.strip()]
            
            # Create structured data
            structured_data = {
                "table_info": {
                    "total_rows": len(table_lines),
                    "total_columns": 1,
                    "headers": ["Content"]
                },
                "data": []
            }
            
            for i, line in enumerate(table_lines):
                structured_data["data"].append({
                    "row": i + 1,
                    "cells": [line.strip()]
                })
            
            return structured_data
            
        except Exception as e:
            logger.error(f"   ❌ Error creating structured data: {e}")
            return {
                "table_info": {"total_rows": 0, "total_columns": 0, "headers": []},
                "data": []
            }
    
    def save_gemini_results(self, table_data: dict, raw_response: str, output_dir: str):
        """
        Save Gemini API extraction results.
        """
        try:
            # Save structured table data
            table_json_path = os.path.join(output_dir, "gemini_table_data.json")
            with open(table_json_path, 'w', encoding='utf-8') as f:
                json.dump(table_data, f, indent=2, ensure_ascii=False)
            logger.info(f"   💾 Table data saved: {table_json_path}")
            
            # Save raw Gemini response
            raw_response_path = os.path.join(output_dir, "gemini_raw_response.txt")
            with open(raw_response_path, 'w', encoding='utf-8') as f:
                f.write(raw_response)
            logger.info(f"   💾 Raw response saved: {raw_response_path}")
            
            # Save summary
            summary_path = os.path.join(output_dir, "gemini_extraction_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("Gemini API Table Extraction Summary\n")
                f.write("=" * 45 + "\n\n")
                f.write(f"Total Rows: {table_data.get('table_info', {}).get('total_rows', 0)}\n")
                f.write(f"Total Columns: {table_data.get('table_info', {}).get('total_columns', 0)}\n")
                f.write(f"Headers: {', '.join(table_data.get('table_info', {}).get('headers', []))}\n")
                f.write(f"Output Directory: {output_dir}\n")
                f.write(f"Timestamp: {__import__('datetime').datetime.now()}\n\n")
                f.write("Table Data Preview:\n")
                f.write("-" * 20 + "\n")
                
                for row_data in table_data.get('data', [])[:5]:  # Show first 5 rows
                    f.write(f"Row {row_data.get('row', 'N/A')}: {row_data.get('cells', [])}\n")
                
                if len(table_data.get('data', [])) > 5:
                    f.write(f"... and {len(table_data.get('data', [])) - 5} more rows\n")
            
            logger.info(f"   📝 Summary saved: {summary_path}")
            
        except Exception as e:
            logger.error(f"   ❌ Error saving results: {e}")
            raise

def main():
    """Main function for Gemini-powered table extraction."""
    parser = argparse.ArgumentParser(description='Gemini API-powered table extraction')
    parser.add_argument('image_path', help='Path to the image containing the table')
    parser.add_argument('--api-key', required=False, help='Gemini API key (optional if GEMINI_API_KEY env var or .env present)')
    parser.add_argument('--output', '-o', default='data/output/gemini_extraction',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        logger.error(f"❌ Image not found: {args.image_path}")
        sys.exit(1)

    # Load .env if present
    load_dotenv()

    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("❌ Gemini API key is required (pass --api-key or set GEMINI_API_KEY in environment/.env)")
        sys.exit(1)
    
    try:
        # Create output directory
        os.makedirs(args.output, exist_ok=True)

        logger.info("🚀 Starting Gemini API Table Extraction...")
        logger.info(f"   Image: {os.path.basename(args.image_path)}")
        logger.info(f"   Output Directory: {args.output}")

        # Initialize Gemini extractor
        extractor = GeminiTableExtractor(api_key)

        # Extract table data
        table_data = extractor.extract_table_with_gemini(args.image_path, args.output)

        # Display results
        logger.info("\n🎉 Gemini API Table Extraction Complete!")
        logger.info(f"   Total Rows: {table_data.get('table_info', {}).get('total_rows', 0)}")
        logger.info(f"   Total Columns: {table_data.get('table_info', {}).get('total_columns', 0)}")
        logger.info(f"   Headers: {', '.join(table_data.get('table_info', {}).get('headers', []))}")
        logger.info(f"   Output Directory: {args.output}")

        # Show sample data
        logger.info("\n📊 Sample Data:")
        for row_data in table_data.get('data', [])[:3]:
            logger.info(f"   Row {row_data.get('row', 'N/A')}: {row_data.get('cells', [])}")

        if len(table_data.get('data', [])) > 3:
            logger.info(f"   ... and {len(table_data.get('data', [])) - 3} more rows")

    except Exception as e:
        logger.error(f"❌ Gemini extraction failed: {e}")
        raise

if __name__ == "__main__":
    main()
