#!/usr/bin/env python3
"""
Configuration file for OCR settings
"""

# Tesseract OCR Configuration
TESSERACT_CONFIG = {
    'language': 'eng',  # English language
    'oem': '3',         # OCR Engine Mode: 3 = Default, based on what is available
    'psm': '7',         # Page Segmentation Mode: 7 = Single text line
    'timeout': 10,      # Timeout in seconds
    'config_string': '--oem 3 --psm 7 -l eng'  # Full config string
}

# Fallback configuration (without language specification)
TESSERACT_FALLBACK_CONFIG = {
    'config_string': '--oem 3 --psm 7'
}

# Image Processing Configuration
IMAGE_PROCESSING = {
    'dpi': 300,                    # PDF to image conversion DPI
    'min_cell_area': 30,          # Minimum cell area for detection
    'row_tolerance': 30,          # Pixel tolerance for grouping cells into rows
    'kernel_sizes': [25, 20, 15], # Kernel sizes for grid detection
    'grid_threshold': 50          # Threshold for grid content detection
}

# Output Configuration
OUTPUT_CONFIG = {
    'save_debug_images': True,     # Save intermediate debug images
    'save_individual_cells': True, # Save individual cell images
    'excel_formatting': True       # Apply Excel formatting
}
