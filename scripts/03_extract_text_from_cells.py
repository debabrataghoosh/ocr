#!/usr/bin/env python3
"""
Step 3: Extract text from identified table cells using OCR.
"""

import os
import sys
import json
import csv
import logging
import numpy as np
import cv2
from PIL import Image
import pytesseract
import argparse
from config import TESSERACT_CONFIG, TESSERACT_FALLBACK_CONFIG

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def identify_table_cells(grid_image_path: str, output_dir: str):
    """
    Step 3: Identify every cell in the grid using multiple detection strategies.
    
    Args:
        grid_image_path: Path to the grid image
        output_dir: Directory to save results
        
    Returns:
        List of cell coordinates (x, y, w, h)
    """
    try:
        logger.info(f"📐 Identifying table cells from: {os.path.basename(grid_image_path)}")
        
        # Load grid image
        grid_image = cv2.imread(grid_image_path, cv2.IMREAD_GRAYSCALE)
        if grid_image is None:
            raise ValueError(f"Could not load grid image: {grid_image_path}")
        
        # Ensure binary image
        _, grid_binary = cv2.threshold(grid_image, 127, 255, cv2.THRESH_BINARY)
        
        # Try multiple detection strategies
        cell_coords = []
        
        # Strategy 1: Contour-based detection
        cell_coords = detect_cells_by_contours(grid_binary)
        
        # Strategy 2: If no cells found, try grid line intersection
        if not cell_coords:
            logger.info("   🔄 Contour detection failed, trying grid line intersection...")
            cell_coords = detect_cells_by_grid_lines(grid_binary)
        
        # Strategy 3: If still no cells, try content-based detection
        if not cell_coords:
            logger.info("   🔄 Grid line detection failed, trying content-based detection...")
            cell_coords = detect_cells_by_content(grid_binary)
        
        if not cell_coords:
            logger.error("   ❌ All cell detection strategies failed")
            return []
        
        # Sort and group cells
        cell_coords = sort_and_group_cells(cell_coords)
        
        logger.info(f"   ✅ Identified {len(cell_coords)} table cells")
        
        # Save cell visualization
        cell_viz_path = os.path.join(output_dir, "05_cell_visualization.png")
        save_cell_visualization(grid_image, cell_coords, cell_viz_path)
        logger.info(f"   💾 Cell visualization saved: {cell_viz_path}")
        
        # Save cell coordinates
        coords_path = os.path.join(output_dir, "cell_coordinates.json")
        with open(coords_path, 'w') as f:
            json.dump(cell_coords, f, indent=2)
        logger.info(f"   💾 Cell coordinates saved: {coords_path}")
        
        return cell_coords
        
    except Exception as e:
        logger.error(f"❌ Error identifying table cells: {e}")
        raise

def detect_cells_by_contours(grid_binary):
    """
    Detect cells using contour detection.
    """
    # Find contours in the grid
    contours, _ = cv2.findContours(grid_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    # Filter contours by area and shape
    valid_cells = []
    min_area = 20  # Very small minimum area
    max_area = grid_binary.shape[0] * grid_binary.shape[1] // 2
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Check area constraints
        if area < min_area or area > max_area:
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by aspect ratio
        aspect_ratio = max(w, h) / max(min(w, h), 1)
        if aspect_ratio > 20:  # Skip extremely long cells
            continue
        
        # Filter by size constraints
        if w < 5 or h < 5:  # Very small minimum
            continue
        if w > grid_binary.shape[1] * 0.95 or h > grid_binary.shape[0] * 0.95:  # Too large
            continue
        
        valid_cells.append((x, y, w, h))
    
    return valid_cells

def detect_cells_by_grid_lines(grid_binary):
    """
    Detect cells using grid line intersection analysis.
    """
    # Find horizontal and vertical lines
    horizontal_lines = find_horizontal_lines(grid_binary)
    vertical_lines = find_vertical_lines(grid_binary)
    
    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        return []
    
    # Create cells from line intersections
    return create_cells_from_grid(horizontal_lines, vertical_lines, grid_binary.shape)

def detect_cells_by_content(grid_binary):
    """
    Detect cells based on content regions.
    """
    # Use connected components to find content regions
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(grid_binary, connectivity=8)
    
    cells = []
    min_area = 50
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Filter by size
            if w >= 10 and h >= 10 and w <= grid_binary.shape[1] * 0.9 and h <= grid_binary.shape[0] * 0.9:
                cells.append((x, y, w, h))
    
    return cells

def sort_and_group_cells(cell_coords):
    """
    Sort and group cells into rows.
    """
    if not cell_coords:
        return []
    
    # Sort cells by position (top to bottom, left to right)
    cell_coords.sort(key=lambda cell: (cell[1], cell[0]))
    
    # Group cells into rows based on y-coordinate
    rows = []
    current_row = []
    row_tolerance = 30  # Increased tolerance
    
    for cell in cell_coords:
        x, y, w, h = cell
        
        if not current_row:
            current_row.append(cell)
        else:
            # Check if this cell belongs to the current row
            first_cell_y = current_row[0][1]
            if abs(y - first_cell_y) <= row_tolerance:
                current_row.append(cell)
            else:
                # Start new row
                if current_row:
                    rows.append(current_row)
                current_row = [cell]
    
    # Add the last row
    if current_row:
        rows.append(current_row)
    
    # Sort cells within each row by x-coordinate
    for row in rows:
        row.sort(key=lambda cell: cell[0])
    
    # Flatten rows back to single list
    final_cells = []
    for row in rows:
        final_cells.extend(row)
    
    return final_cells

def find_horizontal_lines(grid_binary):
    """
    Find horizontal lines in the grid.
    """
    # Use morphological operations to find horizontal lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    horizontal = cv2.morphologyEx(grid_binary, cv2.MORPH_OPEN, kernel)
    
    # Find line positions
    horizontal_lines = []
    for y in range(horizontal.shape[0]):
        if np.sum(horizontal[y, :]) > 0:
            horizontal_lines.append(y)
    
    return horizontal_lines

def find_vertical_lines(grid_binary):
    """
    Find vertical lines in the grid.
    """
    # Use morphological operations to find vertical lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    vertical = cv2.morphologyEx(grid_binary, cv2.MORPH_OPEN, kernel)
    
    # Find line positions
    vertical_lines = []
    for x in range(vertical.shape[1]):
        if np.sum(vertical[:, x]) > 0:
            vertical_lines.append(x)
    
    return vertical_lines

def create_cells_from_grid(horizontal_lines, vertical_lines, image_shape):
    """
    Create cell boundaries from horizontal and vertical line intersections.
    """
    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        logger.warning("   ⚠️ Not enough grid lines found for cell detection")
        return []
    
    # Sort lines
    horizontal_lines.sort()
    vertical_lines.sort()
    
    # Add image boundaries if not present
    if 0 not in horizontal_lines:
        horizontal_lines.insert(0, 0)
    if image_shape[0] - 1 not in horizontal_lines:
        horizontal_lines.append(image_shape[0] - 1)
    
    if 0 not in vertical_lines:
        vertical_lines.insert(0, 0)
    if image_shape[1] - 1 not in vertical_lines:
        vertical_lines.append(image_shape[1] - 1)
    
    # Create cells from line intersections
    cells = []
    min_cell_size = 20  # Minimum cell dimensions
    
    for i in range(len(horizontal_lines) - 1):
        for j in range(len(vertical_lines) - 1):
            y1 = horizontal_lines[i]
            y2 = horizontal_lines[i + 1]
            x1 = vertical_lines[j]
            x2 = vertical_lines[j + 1]
            
            # Calculate cell dimensions
            w = x2 - x1
            h = y2 - y1
            
            # Filter cells by size
            if w >= min_cell_size and h >= min_cell_size:
                cells.append((x1, y1, w, h))
    
    # Sort cells by position (top to bottom, left to right)
    cells.sort(key=lambda cell: (cell[1], cell[0]))
    
    return cells

def save_cell_visualization(grid_image: np.ndarray, cell_coords: list, output_path: str):
    """Save a visualization of detected cells."""
    # Create a copy of the grid image for visualization
    viz_image = cv2.cvtColor(grid_image, cv2.COLOR_GRAY2RGB)
    
    # Draw bounding boxes around each cell
    for i, (x, y, w, h) in enumerate(cell_coords):
        # Draw rectangle
        cv2.rectangle(viz_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add cell number
        cv2.putText(viz_image, str(i + 1), (x + 5, y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Save visualization
    Image.fromarray(viz_image).save(output_path)

def extract_text_from_cells(original_image_path: str, cell_coords: list, output_dir: str):
    """
    Step 4: Perform OCR on each cell individually.
    
    Args:
        original_image_path: Path to the original PNG image
        cell_coords: List of cell coordinates (x, y, w, h)
        output_dir: Directory to save results
        
    Returns:
        List of lists containing extracted text by row
    """
    try:
        logger.info(f"🔤 Extracting text from {len(cell_coords)} cells")
        
        # Load original image
        original_image = np.array(Image.open(original_image_path))
        
        # Check image orientation and try different rotations (same as grid detection)
        height, width = original_image.shape[:2]
        
        logger.info(f"   📐 Original image dimensions: {width} × {height}")
        
        # Try different rotations to find the best orientation
        rotation_options = [
            ("original", original_image, "No rotation"),
            ("clockwise_90", np.rot90(original_image, k=-1), "90° clockwise"),
            ("counter_clockwise_90", np.rot90(original_image, k=1), "90° counter-clockwise"),
            ("180", np.rot90(original_image, k=2), "180° rotation")
        ]
        
        best_rotation = None
        best_rotation_name = "original"
        best_rotation_desc = "No rotation"
        
        # For now, use original orientation since it worked in sectioned approach
        logger.info("   🔄 Testing different image orientations...")
        
        for rotation_name, rotated_img, rotation_desc in rotation_options:
            h, w = rotated_img.shape[:2]
            logger.info(f"      {rotation_desc}: {w} × {h}")
            
            # Check if image is already properly oriented for table detection
            # If it's portrait and has good proportions, use original
            # If it's landscape and very wide, consider rotation
            if rotation_name == "original":
                # Check if original is already good for tables
                if height > width and height/width < 5:  # Portrait but not extremely tall
                    best_rotation = rotated_img
                    best_rotation_name = rotation_name
                    best_rotation_desc = rotation_desc
                    logger.info("   ✅ Image is already properly oriented for table detection")
                    break
                elif width > height and width/height < 5:  # Landscape but not extremely wide
                    best_rotation = rotated_img
                    best_rotation_name = rotation_name
                    best_rotation_desc = rotation_desc
                    logger.info("   ✅ Image is already properly oriented for table detection")
                    break
            elif rotation_name == "clockwise_90":
                # Only use rotation if original is problematic
                if height > width and height/width >= 5:  # Extremely tall portrait
                    best_rotation = rotated_img
                    best_rotation_name = rotation_name
                    best_rotation_desc = rotation_desc
                    logger.info("   🔄 Rotating extremely tall portrait image")
                    break
        
        logger.info(f"   ✅ Using orientation: {best_rotation_desc}")
        
        # Save all rotation options for comparison
        logger.info("   💾 Saving all rotation options for comparison...")
        for rotation_name, rotated_img, rotation_desc in rotation_options:
            rotation_path = os.path.join(output_dir, f"01_rotation_{rotation_name}.png")
            Image.fromarray(rotated_img).save(rotation_path)
            logger.info(f"      💾 {rotation_desc} saved: {rotation_path}")
        
        # Convert to grayscale for better OCR
        if len(best_rotation.shape) == 3:
            gray = cv2.cvtColor(best_rotation, cv2.COLOR_RGB2GRAY)
        else:
            gray = best_rotation
        
        # Group cells by rows (y-coordinate)
        rows = {}
        for x, y, w, h in cell_coords:
            # Group cells within similar y-coordinates (same row)
            row_key = y // 30  # Tolerance of 30 pixels for row grouping
            if row_key not in rows:
                rows[row_key] = []
            rows[row_key].append((x, y, w, h))
        
        # Sort each row by x-coordinate (left to right)
        for row_key in rows:
            rows[row_key].sort(key=lambda coord: coord[0])
        
        # Extract text from each cell
        table_data = []
        cell_details = []
        
        for row_key in sorted(rows.keys()):
            row_cells = rows[row_key]
            row_data = []
            
            logger.info(f"      Processing row {row_key + 1} with {len(row_cells)} cells")
            
            for i, (x, y, w, h) in enumerate(row_cells):
                # Crop the cell from the original image
                cell_img = gray[y:y+h, x:x+w]
                
                # Save individual cell images for debugging
                cell_img_path = os.path.join(output_dir, f"cell_{row_key}_{i:02d}.png")
                Image.fromarray(cell_img).save(cell_img_path)
                
                # Preprocess the cell image for better OCR
                _, cell_binary = cv2.threshold(cell_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Save binary cell image
                cell_binary_path = os.path.join(output_dir, f"cell_{row_key}_{i:02d}_binary.png")
                Image.fromarray(cell_binary).save(cell_binary_path)
                
                # Run Tesseract OCR on the cell
                try:
                    # Use English language explicitly for better accuracy
                    text = pytesseract.image_to_string(
                        cell_img, 
                        config=TESSERACT_CONFIG['config_string'],  # English language, PSM 7 for single line
                        timeout=TESSERACT_CONFIG['timeout']
                    )
                    
                    # If English fails, try without language specification as fallback
                    if not text.strip():
                        text = pytesseract.image_to_string(
                            cell_img, 
                            config=TESSERACT_FALLBACK_CONFIG['config_string'],  # Fallback without language
                            timeout=TESSERACT_CONFIG['timeout']
                        )
                        
                except Exception as e:
                    logger.warning(f"      ⚠️ OCR failed for cell {i+1}: {e}")
                    text = ""
                
                # Clean the extracted text
                cleaned_text = text.strip()
                if not cleaned_text:
                    cleaned_text = ""  # Empty cell
                
                row_data.append(cleaned_text)
                
                # Store cell details
                cell_details.append({
                    'row': row_key + 1,
                    'col': i + 1,
                    'x': x, 'y': y, 'width': w, 'height': h,
                    'text': cleaned_text,
                    'cell_image': cell_img_path,
                    'binary_image': cell_binary_path
                })
                
                logger.info(f"         Cell {i+1}: '{cleaned_text}' (x={x}, y={y}, w={w}, h={h})")
            
            if row_data:  # Only add non-empty rows
                table_data.append(row_data)
        
        # Save extracted text data
        text_data_path = os.path.join(output_dir, "extracted_text.json")
        with open(text_data_path, 'w') as f:
            json.dump({
                'total_rows': len(table_data),
                'total_cells': len(cell_coords),
                'table_data': table_data,
                'cell_details': cell_details
            }, f, indent=2)
        
        # Save table data as CSV for easy viewing
        csv_path = os.path.join(output_dir, "extracted_text.csv")
        import pandas as pd
        df = pd.DataFrame(table_data)
        df.to_csv(csv_path, index=False, header=False)
        
        logger.info(f"   ✅ Extracted text from {len(table_data)} rows")
        logger.info(f"   💾 Text data saved: {text_data_path}")
        logger.info(f"   💾 CSV data saved: {csv_path}")
        
        return table_data
        
    except Exception as e:
        logger.error(f"❌ Error extracting text from cells: {e}")
        raise

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Identify cells and extract text from table images')
    parser.add_argument('grid_image', help='Path to the grid image')
    parser.add_argument('original_image', help='Path to the original PNG image')
    parser.add_argument('--output', '-o', default='data/output/text_extraction', 
                       help='Output directory for results (default: data/output/text_extraction)')
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.grid_image):
        logger.error(f"❌ Grid image not found: {args.grid_image}")
        sys.exit(1)
    
    if not os.path.exists(args.original_image):
        logger.error(f"❌ Original image not found: {args.original_image}")
        sys.exit(1)
    
    try:
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        # Step 3: Identify table cells
        cell_coords = identify_table_cells(args.grid_image, args.output)
        
        if not cell_coords:
            logger.error("❌ No cells identified, cannot proceed with text extraction")
            sys.exit(1)
        
        # Step 4: Extract text from cells
        table_data = extract_text_from_cells(args.original_image, cell_coords, args.output)
        
        print(f"\n🎯 Text Extraction Summary:")
        print(f"   Grid Image: {args.grid_image}")
        print(f"   Original Image: {args.original_image}")
        print(f"   Cells Identified: {len(cell_coords)}")
        print(f"   Rows Extracted: {len(table_data)}")
        print(f"   Output Directory: {args.output}")
        
        # Show sample data
        if table_data:
            print(f"\n📊 Sample Extracted Data:")
            for i, row in enumerate(table_data[:3]):  # Show first 3 rows
                print(f"   Row {i+1}: {row}")
            
    except Exception as e:
        logger.error(f"❌ Text extraction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
