#!/usr/bin/env python3
"""
Configuration file for OCR settings
"""

# EasyOCR Configuration
EASYOCR_CONFIG = {
    'languages': ['en'],  # English language
    'gpu': False,         # Use CPU for compatibility
    'model_storage_directory': None,  # Use default model directory
    'user_network_directory': None,   # Use default user network directory
    'detail': 0,          # No detailed output (just text)
    'paragraph': False,   # Single line mode
    'batch_size': 1       # Process one image at a time
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
