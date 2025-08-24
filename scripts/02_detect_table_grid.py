#!/usr/bin/env python3
"""
Script 2: Table Grid Detection
Detects table grids in PNG images using OpenCV morphological operations.
"""

import os
import sys
import logging
import cv2
import numpy as np
import argparse
from PIL import Image

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_image_for_line_detection(image_path: str):
    """
    Step 1: Prepare image for line detection.
    
    Args:
        image_path: Path to the PNG image
        
    Returns:
        Prepared image array
    """
    logger.info(f"🔧 Preparing image for line detection: {os.path.basename(image_path)}")
    
    # Load image
    img_array = np.array(Image.open(image_path))
    
    # Check image orientation and try different rotations
    height, width = img_array.shape[:2]
    
    logger.info(f"   📐 Original image dimensions: {width} × {height}")
    
    # Try different rotations to find the best orientation
    rotation_options = [
        ("original", img_array, "No rotation"),
        ("clockwise_90", np.rot90(img_array, k=-1), "90° clockwise"),
        ("counter_clockwise_90", np.rot90(img_array, k=1), "90° counter-clockwise"),
        ("180", np.rot90(img_array, k=2), "180° rotation")
    ]
    
    best_rotation = None
    best_rotation_name = "original"
    best_rotation_desc = "No rotation"
    
    # For now, let's use the original orientation since the sectioned approach worked well
    # But save all rotation options for debugging
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
    
    # Convert to grayscale
    if len(best_rotation.shape) == 3:
        gray = cv2.cvtColor(best_rotation, cv2.COLOR_RGB2GRAY)
    else:
        gray = best_rotation
    
    # Apply adaptive thresholding for clean black-and-white image
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Invert the image (make lines white, background black)
    inverted = cv2.bitwise_not(binary)
    
    # Thicken the lines using dilation
    kernel = np.ones((2, 2), np.uint8)
    thickened = cv2.dilate(inverted, kernel, iterations=1)
    
    logger.info("   ✅ Image prepared for line detection")
    return thickened, best_rotation, rotation_options  # Return processed, best rotation, and all options

def break_image_into_sections(image: np.ndarray):
    """
    Break the image into adaptive sections based on content analysis.
    """
    height, width = image.shape
    
    # Analyze image content to determine optimal section sizes
    # Look for natural breaks in content (empty rows, text density changes)
    
    # Calculate row-wise content density
    row_density = []
    for y in range(0, height, 10):  # Sample every 10th row
        row = image[y:min(y+10, height), :]
        density = np.sum(row > 0) / (row.shape[0] * row.shape[1])
        row_density.append((y, density))
    
    # Find natural break points (low density areas)
    break_points = []
    for i in range(1, len(row_density)):
        if abs(row_density[i][1] - row_density[i-1][1]) > 0.1:  # Significant density change
            break_points.append(row_density[i][0])
    
    # Add start and end points
    break_points = [0] + break_points + [height]
    
    # Create sections based on natural breaks
    sections = []
    min_section_height = 200  # Minimum section height
    
    for i in range(len(break_points) - 1):
        start_y = break_points[i]
        end_y = break_points[i + 1]
        section_height = end_y - start_y
        
        if section_height >= min_section_height:
            section = image[start_y:end_y, 0:width]
            sections.append((start_y, end_y, section))
    
    # If no natural breaks found, use adaptive sectioning
    if len(sections) <= 1:
        # Use content-aware adaptive sectioning
        sections = adaptive_sectioning(image)
    
    # If still no sections, use fallback sectioning
    if len(sections) <= 1:
        sections = fallback_sectioning(image)
    
    logger.info(f"   📐 Created {len(sections)} adaptive sections for analysis")
    return sections

def adaptive_sectioning(image: np.ndarray):
    """
    Create adaptive sections based on content analysis.
    """
    height, width = image.shape
    
    # Analyze vertical content distribution
    vertical_profile = np.sum(image > 0, axis=1)
    
    # Find regions with significant content
    content_threshold = np.mean(vertical_profile) * 0.3
    content_regions = vertical_profile > content_threshold
    
    # Find boundaries of content regions
    boundaries = []
    in_content = False
    start_y = 0
    
    for y in range(height):
        if content_regions[y] and not in_content:
            start_y = y
            in_content = True
        elif not content_regions[y] and in_content:
            if y - start_y > 100:  # Minimum region size
                boundaries.append((start_y, y))
            in_content = False
    
    # Add final region if still in content
    if in_content and height - start_y > 100:
        boundaries.append((start_y, height))
    
    # Create sections with some overlap
    sections = []
    overlap = 50
    
    for start_y, end_y in boundaries:
        # Add overlap to previous section
        if sections:
            prev_start, prev_end, prev_section = sections[-1]
            # Extend previous section with overlap
            extended_end = min(prev_end + overlap, end_y)
            extended_section = np.vstack([prev_section, image[prev_end:extended_end, :]])
            sections[-1] = (prev_start, extended_end, extended_section)
        
        # Create new section
        section = image[start_y:end_y, 0:width]
        sections.append((start_y, end_y, section))
    
    return sections

def fallback_sectioning(image: np.ndarray):
    """
    Fallback sectioning when other methods fail.
    """
    height, width = image.shape
    
    # Use simple overlapping sections as fallback
    section_height = min(height // 3, 400)  # Smaller sections for better detection
    overlap = 50
    
    sections = []
    
    for y in range(0, height, section_height - overlap):
        end_y = min(y + section_height, height)
        if end_y - y >= 100:  # Minimum section height
            section = image[y:end_y, 0:width]
            sections.append((y, end_y, section))
    
    # Ensure we have at least one section
    if not sections:
        # Create one section covering the entire image
        section = image[0:height, 0:width]
        sections.append((0, height, section))
    
    return sections

def detect_grid_in_section(section: np.ndarray):
    """
    Detect grid lines using multiple approaches and combine results.
    """
    height, width = section.shape
    
    # Multiple detection strategies
    grids = []
    
    # Strategy 1: Morphological operations with multiple kernel sizes
    kernel_sizes = [
        (20, 1), (1, 20),    # Standard tables
        (15, 1), (1, 15),    # Small tables
        (30, 1), (1, 30),    # Large tables
        (40, 1), (1, 40),    # Very large tables
    ]
    
    for h_kernel, v_kernel in kernel_sizes:
        if h_kernel <= width and v_kernel <= height:
            grid = detect_with_kernels(section, h_kernel, v_kernel)
            if grid is not None:
                grids.append(grid)
    
    # Strategy 2: Edge detection approach
    edge_grid = detect_with_edges(section)
    if edge_grid is not None:
        grids.append(edge_grid)
    
    # Strategy 3: Line detection approach
    line_grid = detect_with_lines(section)
    if line_grid is not None:
        grids.append(line_grid)
    
    # Combine all detected grids
    if not grids:
        return None
    
    # Use weighted combination based on grid quality
    combined_grid = combine_grids_intelligently(grids, section)
    
    return combined_grid

def detect_with_kernels(section: np.ndarray, h_kernel: int, v_kernel: int):
    """
    Detect grid using morphological operations with specific kernels.
    """
    try:
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
        if grid_content > 50:  # Higher threshold for quality
            return grid
        
    except Exception as e:
        logger.debug(f"      Kernel detection failed: {e}")
    
    return None

def detect_with_edges(section: np.ndarray):
    """
    Detect grid using edge detection approach.
    """
    try:
        # Apply Canny edge detection
        edges = cv2.Canny(section, 50, 150)
        
        # Use morphological operations to connect edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Convert to binary
        grid = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)[1]
        
        # Check quality
        if np.sum(grid > 0) > 100:
            return grid
            
    except Exception as e:
        logger.debug(f"      Edge detection failed: {e}")
    
    return None

def detect_with_lines(section: np.ndarray):
    """
    Detect grid using Hough line detection.
    """
    try:
        # Apply Hough line detection
        lines = cv2.HoughLinesP(section, 1, np.pi/180, threshold=50, 
                               minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            # Create blank image for lines
            grid = np.zeros_like(section)
            
            # Draw detected lines
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(grid, (x1, y1), (x2, y2), 255, 2)
            
            # Check quality
            if np.sum(grid > 0) > 80:
                return grid
                
    except Exception as e:
        logger.debug(f"      Line detection failed: {e}")
    
    return None

def clean_combined_grid(grid: np.ndarray):
    """
    Clean and improve the combined grid for better cell segmentation.
    """
    # Remove noise and small artifacts
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(grid, cv2.MORPH_OPEN, kernel)
    
    # Close gaps in grid lines
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    # Ensure grid lines are continuous
    # Find horizontal and vertical components
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    
    horizontal_lines = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, vertical_kernel)
    
    # Recombine for cleaner grid
    cleaned = cv2.bitwise_or(horizontal_lines, vertical_lines)
    
    # Final cleanup
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def combine_grids_intelligently(grids: list, section: np.ndarray):
    """
    Intelligently combine multiple detected grids.
    """
    if len(grids) == 1:
        return grids[0]
    
    # Calculate quality score for each grid
    grid_scores = []
    for grid in grids:
        # Score based on content density and structure
        density = np.sum(grid > 0) / (grid.shape[0] * grid.shape[1])
        
        # Check for horizontal and vertical structure
        h_structure = np.sum(np.sum(grid, axis=1) > 0) / grid.shape[0]
        v_structure = np.sum(np.sum(grid, axis=0) > 0) / grid.shape[1]
        
        # Combined score
        score = density * (h_structure + v_structure)
        grid_scores.append((score, grid))
    
    # Sort by score
    grid_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Use weighted combination of top grids
    top_grids = grid_scores[:min(3, len(grid_scores))]
    
    # Create weighted combination
    combined = np.zeros_like(section, dtype=np.float32)
    total_weight = 0
    
    for i, (score, grid) in enumerate(top_grids):
        weight = score * (0.7 ** i)  # Decreasing weight for lower ranked grids
        combined += weight * grid.astype(np.float32)
        total_weight += weight
    
    # Normalize and convert to uint8
    if total_weight > 0:
        combined = (combined / total_weight).astype(np.uint8)
    
    return combined

def detect_table_grid(image_path: str, output_dir: str):
    """
    Step 2: Detect table grid lines by breaking into sections.
    
    Args:
        image_path: Path to the PNG image
        output_dir: Directory to save results
        
    Returns:
        Path to the detected grid image
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"🔍 Detecting table grid in: {os.path.basename(image_path)}")
        
        # Step 1: Prepare image
        prepared_image, original_rotated_image, rotation_options = prepare_image_for_line_detection(image_path)
        
        # Save prepared image
        prepared_path = os.path.join(output_dir, "01_prepared_image.png")
        Image.fromarray(prepared_image).save(prepared_path)
        logger.info(f"   💾 Prepared image saved: {prepared_path}")
        
        # Save all rotation options for comparison
        logger.info("   💾 Saving all rotation options for comparison...")
        for rotation_name, rotated_img, rotation_desc in rotation_options:
            rotation_path = os.path.join(output_dir, f"01_rotation_{rotation_name}.png")
            Image.fromarray(rotated_img).save(rotation_path)
            logger.info(f"      💾 {rotation_desc} saved: {rotation_path}")
        
        # Save original rotated image for debugging
        if original_rotated_image is not None:
            original_rotated_path = os.path.join(output_dir, "01_original_rotated_image.png")
            Image.fromarray(original_rotated_image).save(original_rotated_path)
            logger.info(f"   💾 Original rotated image saved: {original_rotated_path}")
        
        # Step 2: Break into sections
        sections = break_image_into_sections(prepared_image)
        
        all_grids = []
        section_info = []
        
        for i, (y_start, y_end, section) in enumerate(sections):
            logger.info(f"      Processing section {i+1}/{len(sections)} (y={y_start}-{y_end})")
            
            # Save section for debugging
            section_path = os.path.join(output_dir, f"02_section_{i+1}.png")
            Image.fromarray(section).save(section_path)
            
            # Detect grid in section
            section_grid = detect_grid_in_section(section)
            
            if section_grid is not None:
                all_grids.append((y_start, y_end, section_grid))
                section_info.append(f"Section {i+1}: Grid detected (y={y_start}-{y_end})")
                logger.info(f"      ✅ Grid detected in section {i+1}")
                
                # Save grid for debugging
                grid_path = os.path.join(output_dir, f"03_grid_section_{i+1}.png")
                Image.fromarray(section_grid).save(grid_path)
            else:
                section_info.append(f"Section {i+1}: No grid (y={y_start}-{y_end})")
                logger.info(f"      ⚠️ No grid in section {i+1}")
        
        if not all_grids:
            logger.warning("   ⚠️ No table grids detected in any section")
            return None
        
        # Combine all detected grids intelligently
        height, width = prepared_image.shape
        combined_grid = np.zeros((height, width), dtype=np.uint8)
        
        # Sort grids by quality (more content = better quality)
        grid_quality = []
        for y_start, y_end, grid in all_grids:
            quality = np.sum(grid > 0)
            grid_quality.append((quality, y_start, y_end, grid))
        
        # Sort by quality (highest first)
        grid_quality.sort(key=lambda x: x[0], reverse=True)
        
        # Combine grids with priority for higher quality ones
        for quality, y_start, y_end, grid in grid_quality:
            grid_height, grid_width = grid.shape
            
            # Check if grid dimensions match the expected section
            if grid_height == (y_end - y_start) and grid_width == width:
                # Place grid in correct position
                combined_grid[y_start:y_end, 0:width] = cv2.bitwise_or(
                    combined_grid[y_start:y_end, 0:width], grid
                )
            else:
                logger.warning(f"      ⚠️ Grid dimensions mismatch in section: expected {y_end-y_start}×{width}, got {grid_height}×{grid_width}")
                # Resize grid to match section dimensions
                resized_grid = cv2.resize(grid, (width, y_end - y_start))
                combined_grid[y_start:y_end, 0:width] = cv2.bitwise_or(
                    combined_grid[y_start:y_end, 0:width], resized_grid
                )
        
        # Post-process the combined grid for cleaner cell boundaries
        combined_grid = clean_combined_grid(combined_grid)
        
        # Save combined grid
        combined_path = os.path.join(output_dir, "04_combined_grid.png")
        Image.fromarray(combined_grid).save(combined_path)
        
        # Save detection info
        info_path = os.path.join(output_dir, "grid_detection_info.txt")
        with open(info_path, 'w') as f:
            f.write(f"Image: {image_path}\n")
            f.write(f"Total Sections: {len(sections)}\n")
            f.write(f"Grids Detected: {len(all_grids)}\n\n")
            f.write("Section Analysis:\n")
            for info in section_info:
                f.write(f"  {info}\n")
            f.write(f"\nCombined Grid: {combined_path}\n")
        
        logger.info(f"   ✅ Table grids detected in {len(all_grids)} sections")
        logger.info(f"   💾 Combined grid saved: {combined_path}")
        logger.info(f"   📝 Detection info saved: {info_path}")
        
        return combined_path
        
    except Exception as e:
        logger.error(f"❌ Error detecting table grid: {e}")
        raise

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Detect table grids in PNG images')
    parser.add_argument('image_path', help='Path to the PNG image')
    parser.add_argument('--output', '-o', default='data/output/grid_detection', 
                       help='Output directory for results (default: data/output/grid_detection)')
    
    args = parser.parse_args()
    
    # Check if image file exists
    if not os.path.exists(args.image_path):
        logger.error(f"❌ Image file not found: {args.image_path}")
        sys.exit(1)
    
    try:
        # Detect table grid
        grid_path = detect_table_grid(args.image_path, args.output)
        
        if grid_path:
            print(f"\n🎯 Grid Detection Summary:")
            print(f"   Image: {args.image_path}")
            print(f"   Grid Detected: Yes")
            print(f"   Output Directory: {args.output}")
            print(f"   Combined Grid: {os.path.basename(grid_path)}")
        else:
            print(f"\n⚠️ No table grid detected in the image")
            
    except Exception as e:
        logger.error(f"❌ Grid detection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
