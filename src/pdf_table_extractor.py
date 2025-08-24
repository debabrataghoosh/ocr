#!/usr/bin/env python3
"""
PDF Table Extractor with Advanced Image Processing and Line Detection
Uses OpenCV for grid detection and Tesseract for cell-by-cell OCR
"""

import os
import cv2
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional, Union
import pytesseract
from pdf2image import convert_from_path
import json
from PIL import Image # Added for debug image saving

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFTableExtractor:
    """Extract tables from PDF using advanced image processing and line detection."""
    
    def __init__(self):
        """Initialize the PDF table extractor."""
        self.extracted_tables = []
        self.cell_coordinates = []
        
        # Check if required libraries are available
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import cv2
            logger.info("✅ OpenCV is available")
        except ImportError:
            raise Exception("OpenCV is required. Install with: pip install opencv-python")
        
        try:
            import pytesseract
            logger.info("✅ Tesseract OCR is available")
        except ImportError:
            raise Exception("Tesseract is required. Install with: pip install pytesseract")
        
        try:
            from pdf2image import convert_from_path
            logger.info("✅ pdf2image is available")
        except ImportError:
            raise Exception("pdf2image is required. Install with: pip install pdf2image")
    
    def extract_tables_from_pdf(self, pdf_path: str, pages: str = "all") -> List[pd.DataFrame]:
        """
        Extract tables from PDF using the new image processing approach.
        """
        logger.info(f"🚀 Starting advanced table extraction from: {os.path.basename(pdf_path)}")
        logger.info("=" * 70)
        
        # Convert PDF to images
        images = self._convert_pdf_to_images(pdf_path, pages)
        
        all_tables = []
        
        for i, image in enumerate(images):
            logger.info(f"\n📄 Processing page {i+1}/{len(images)}")
            logger.info("-" * 50)
            
            # Step 1: Image preparation for line detection
            processed_image = self._prepare_image_for_line_detection(image)
            
            # Step 2: Detect horizontal and vertical grid lines
            grid_image = self._detect_table_grid(processed_image)
            
            if grid_image is None:
                logger.warning(f"   ⚠️ No table grid detected on page {i+1}")
                continue
            
            # Step 3: Identify every cell in the grid
            cell_coords = self._identify_table_cells(grid_image)
            
            if not cell_coords:
                logger.warning(f"   ⚠️ No table cells detected on page {i+1}")
                continue
            
            # Step 4: Perform OCR on each cell individually
            table_data = self._extract_text_from_cells(image, cell_coords)
            
            if table_data:
                # Step 5: Rebuild the table and save
                df = self._rebuild_table(table_data, cell_coords)
                all_tables.append(df)
                
                logger.info(f"   ✅ Extracted table: {df.shape[0]} rows × {df.shape[1]} columns")
                logger.info(f"   📊 Total cells: {len(cell_coords)}")
        
        self.extracted_tables = all_tables
        logger.info(f"\n🎉 Extraction completed successfully!")
        logger.info(f"📊 Total tables extracted: {len(all_tables)}")
        
        return all_tables
    
    def _convert_pdf_to_images(self, pdf_path: str, pages: str) -> List:
        """Convert PDF pages to images for processing."""
        try:
            logger.info("   🔄 Converting PDF to images...")
            
            # Determine which pages to process
            if pages == "all":
                page_numbers = None
            else:
                page_numbers = [int(p) - 1 for p in pages.split(',')]
            
            # Convert PDF to images with high DPI for better quality
            images = convert_from_path(pdf_path, dpi=300, first_page=page_numbers[0] if page_numbers else None, last_page=page_numbers[-1] if page_numbers else None)
            
            logger.info(f"   ✅ Converted {len(images)} pages to images")
            return images
            
        except Exception as e:
            logger.error(f"   ❌ Error converting PDF to images: {e}")
            raise
    
    def _prepare_image_for_line_detection(self, image) -> np.ndarray:
        """
        Step 1: Focus on Image Preparation for Line Detection
        Create a perfect black-and-white image where only table grid lines are visible.
        """
        logger.info("   🔧 Step 1: Preparing image for line detection...")
        
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply adaptive thresholding for clean black-and-white image
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Invert the image (make lines white, background black)
        inverted = cv2.bitwise_not(binary)
        
        # Thicken the lines using dilation (crucial for connecting small breaks)
        kernel = np.ones((2, 2), np.uint8)
        thickened = cv2.dilate(inverted, kernel, iterations=1)
        
        # Save debug image
        self._save_debug_image(thickened, "01_prepared_image")
        
        logger.info("   ✅ Image prepared for line detection")
        return thickened
    
    def _save_debug_image(self, image: np.ndarray, name: str):
        """Save debug images to help visualize the processing steps."""
        try:
            debug_dir = "data/output/debug"
            os.makedirs(debug_dir, exist_ok=True)
            
            # Convert to PIL image for saving
            if len(image.shape) == 3:
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray(image, mode='L')
            
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.png"
            filepath = os.path.join(debug_dir, filename)
            
            pil_image.save(filepath)
            logger.info(f"      💾 Debug image saved: {filepath}")
            
        except Exception as e:
            logger.warning(f"      ⚠️ Could not save debug image: {e}")
    
    def _detect_table_grid(self, processed_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Step 2: Detect the Horizontal and Vertical Grid Lines
        Find all lines that make up the table grid by breaking into sections.
        """
        logger.info("   🔍 Step 2: Detecting table grid lines in sections...")
        
        # Create a copy of the processed image
        img = processed_image.copy()
        height, width = img.shape
        
        # Break the image into sections for better table detection
        sections = self._break_image_into_sections(img)
        
        all_grids = []
        
        for i, section in enumerate(sections):
            logger.info(f"      Processing section {i+1}/{len(sections)}")
            
            # Save section for debugging
            self._save_debug_image(section, f"02_section_{i+1}")
            
            # Try different kernel sizes for each section
            section_grid = self._detect_grid_in_section(section)
            
            if section_grid is not None:
                all_grids.append(section_grid)
                logger.info(f"      ✅ Grid detected in section {i+1}")
                
                # Save grid for debugging
                self._save_debug_image(section_grid, f"03_grid_section_{i+1}")
            else:
                logger.info(f"      ⚠️ No grid in section {i+1}")
        
        if not all_grids:
            logger.warning("   ⚠️ No table grids detected in any section")
            return None
        
        # Combine all detected grids
        combined_grid = np.zeros_like(img)
        for grid in all_grids:
            combined_grid = cv2.bitwise_or(combined_grid, grid)
        
        # Save combined grid for debugging
        self._save_debug_image(combined_grid, "04_combined_grid")
        
        logger.info(f"   ✅ Table grids detected in {len(all_grids)} sections")
        return combined_grid
    
    def _break_image_into_sections(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Break the image into sections for better table detection.
        This helps identify multiple tables or table-like structures.
        """
        height, width = image.shape
        
        # Define section sizes - adaptive based on image dimensions
        section_height = min(height // 3, 800)  # Max 800px per section
        section_width = width
        
        sections = []
        
        # Create overlapping sections for better coverage
        overlap = 100  # 100 pixel overlap between sections
        
        for y in range(0, height, section_height - overlap):
            # Ensure we don't go beyond image boundaries
            end_y = min(y + section_height, height)
            if end_y - y < 200:  # Skip very small sections
                continue
                
            section = image[y:end_y, 0:width]
            sections.append(section)
            
            # If this is the last section and it's small, merge with previous
            if len(sections) > 1 and (end_y - y) < section_height // 2:
                sections[-2] = np.vstack([sections[-2], section])
                sections.pop()
        
        logger.info(f"      Created {len(sections)} sections for analysis")
        return sections
    
    def _detect_grid_in_section(self, section: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect grid lines in a specific section using multiple kernel sizes.
        """
        # Try different kernel sizes for different table types
        kernel_sizes = [
            (25, 1), (1, 25),    # Medium tables
            (15, 1), (1, 15),    # Small tables
            (35, 1), (1, 35),    # Large tables
        ]
        
        best_grid = None
        max_grid_content = 0
        
        for h_kernel, v_kernel in kernel_sizes:
            # Create kernels
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel))
            
            # Detect lines
            horizontal_lines = cv2.morphologyEx(section, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(section, cv2.MORPH_OPEN, vertical_kernel)
            
            # Combine lines
            grid = cv2.add(horizontal_lines, vertical_lines)
            
            # Clean up the grid
            kernel_clean = np.ones((3, 3), np.uint8)
            grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel_clean)
            
            # Check grid quality
            grid_content = np.sum(grid > 0)
            
            if grid_content > max_grid_content and grid_content > 30:  # Minimum threshold
                max_grid_content = grid_content
                best_grid = grid
        
        return best_grid
    
    def _identify_table_cells(self, grid_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Step 3: Identify Every Cell in the Grid
        Find the exact location of every single cell using contours.
        """
        logger.info("   📐 Step 3: Identifying table cells...")
        
        # Find contours in the grid image
        contours, _ = cv2.findContours(grid_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("   ⚠️ No contours found in grid image")
            return []
        
        # Filter contours by area to remove very small ones - More flexible
        min_area = 30  # Reduced from 50 to 30 for better cell detection
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        if not valid_contours:
            logger.warning("   ⚠️ No valid cell contours found")
            return []
        
        # Get bounding rectangles for each contour
        cell_coords = []
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cell_coords.append((x, y, w, h))
        
        # Sort contours from top-to-bottom and left-to-right
        # This is crucial for rebuilding the table in correct sequence
        cell_coords.sort(key=lambda coord: (coord[1], coord[0]))  # Sort by y, then x
        
        logger.info(f"   ✅ Identified {len(cell_coords)} table cells")
        
        # Debug: Log cell coordinates for better understanding
        for i, (x, y, w, h) in enumerate(cell_coords):
            logger.info(f"      Cell {i+1}: x={x}, y={y}, w={w}, h={h}, area={w*h}")
        
        return cell_coords
    
    def _extract_text_from_cells(self, original_image, cell_coords: List[Tuple[int, int, int, int]]) -> List[List[str]]:
        """
        Step 4: Perform OCR on Each Cell Individually
        Run Tesseract on one tiny cell at a time for accurate text extraction.
        """
        logger.info("   🔤 Step 4: Extracting text from individual cells...")
        
        # Convert PIL image to numpy array
        img_array = np.array(original_image)
        
        # Convert to grayscale for better OCR
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Group cells by rows (y-coordinate) - More flexible grouping
        rows = {}
        for x, y, w, h in cell_coords:
            # Group cells within similar y-coordinates (same row) - Increased tolerance
            row_key = y // 30  # Increased tolerance from 20 to 30 pixels for better row grouping
            if row_key not in rows:
                rows[row_key] = []
            rows[row_key].append((x, y, w, h))
        
        # Sort each row by x-coordinate (left to right)
        for row_key in rows:
            rows[row_key].sort(key=lambda coord: coord[0])
        
        # Extract text from each cell
        table_data = []
        for row_key in sorted(rows.keys()):
            row_cells = rows[row_key]
            row_data = []
            
            logger.info(f"      Processing row {row_key + 1} with {len(row_cells)} cells")
            
            for i, (x, y, w, h) in enumerate(row_cells):
                # Crop the cell from the original image
                cell_img = gray[y:y+h, x:x+w]
                
                # Preprocess the cell image for better OCR
                # Apply threshold to make text more prominent
                _, cell_binary = cv2.threshold(cell_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Run Tesseract OCR on the cell
                try:
                    # Use English language explicitly for better accuracy
                    text = pytesseract.image_to_string(
                        cell_binary, 
                        config='--oem 3 --psm 7 -l eng',  # English language, PSM 7 for single line
                        timeout=10
                    )
                    
                    # If English fails, try without language specification as fallback
                    if not text.strip():
                        text = pytesseract.image_to_string(
                            cell_binary, 
                            config='--oem 3 --psm 7',  # Fallback without language
                            timeout=10
                        )
                        
                except Exception as e:
                    logger.warning(f"      ⚠️ OCR failed for cell {i+1}: {e}")
                    text = ""
                
                # Clean the extracted text
                cleaned_text = text.strip()
                if not cleaned_text:
                    cleaned_text = ""  # Empty cell
                
                row_data.append(cleaned_text)
                logger.info(f"         Cell {i+1}: '{cleaned_text}' (x={x}, y={y}, w={w}, h={h})")
            
            if row_data:  # Only add non-empty rows
                table_data.append(row_data)
        
        logger.info(f"   ✅ Extracted text from {len(table_data)} rows")
        return table_data
    
    def _rebuild_table(self, table_data: List[List[str]], cell_coords: List[Tuple[int, int, int, int]]) -> pd.DataFrame:
        """
        Step 5: Rebuild the Table and Save
        Assemble the results into a structured DataFrame.
        """
        logger.info("   🏗️ Step 5: Rebuilding table structure...")
        
        if not table_data:
            return pd.DataFrame()
        
        # Find the maximum number of columns
        max_cols = max(len(row) for row in table_data)
        
        # Pad rows with fewer columns
        padded_data = []
        for row in table_data:
            padded_row = row + [""] * (max_cols - len(row))
            padded_data.append(padded_row)
        
        # Create DataFrame
        df = pd.DataFrame(padded_data)
        
        # Clean the data
        df = self._clean_table_data(df)
        
        logger.info("   ✅ Table structure rebuilt successfully")
        return df
    
    def _clean_table_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and format the extracted table data."""
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean cell values
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace('nan', '')
            df[col] = df[col].replace('None', '')
        
        return df
    
    def export_to_excel(self, output_path: str) -> str:
        """Export extracted tables to Excel."""
        try:
            if not self.extracted_tables:
                raise Exception("No tables to export. Run extraction first.")
            
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            logger.info(f"💾 Exporting {len(self.extracted_tables)} tables to Excel...")
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for i, table in enumerate(self.extracted_tables):
                    sheet_name = f"Table_{i+1}"
                    
                    # Export table to Excel sheet
                    table.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                    
                    # Get the worksheet for formatting
                    worksheet = writer.sheets[sheet_name]
                    
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
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
                    
                    # Add borders to make it look like a table
                    from openpyxl.styles import Border, Side
                    thin_border = Border(
                        left=Side(style='thin'),
                        right=Side(style='thin'),
                        top=Side(style='thin'),
                        bottom=Side(style='thin')
                    )
                    
                    # Apply borders to all cells
                    for row in worksheet.iter_rows():
                        for cell in row:
                            cell.border = thin_border
                    
                    logger.info(f"   ✅ Sheet '{sheet_name}': {table.shape[0]} rows × {table.shape[1]} columns")
            
            logger.info(f"💾 Excel file saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Error exporting to Excel: {e}")
            raise
    
    def get_table_summary(self) -> Dict:
        """Get summary of extracted tables."""
        summary = {
            'total_tables': len(self.extracted_tables),
            'tables': []
        }
        
        for i, table in enumerate(self.extracted_tables):
            table_info = {
                'table_number': i + 1,
                'dimensions': f"{table.shape[0]} rows × {table.shape[1]} columns",
                'total_cells': table.shape[0] * table.shape[1],
                'non_empty_cells': table.notna().sum().sum(),
                'sample_data': table.head(3).to_dict('records') if not table.empty else []
            }
            summary['tables'].append(table_info)
        
        return summary
