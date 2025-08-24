# 🔧 **Individual Scripts for Table Extraction**

This directory contains separate, focused scripts for each step of the table extraction process. You can run them individually for debugging or run the complete pipeline.

## 📋 **Available Scripts**

### **1. `01_pdf_to_png.py` - PDF to PNG Conversion**
**What it does:** Converts PDF files to high-quality PNG images for processing.

**Usage:**
```bash
python3 scripts/01_pdf_to_png.py input.pdf --output data/output/png --dpi 300
```

**Arguments:**
- `input.pdf` - Path to the PDF file (required)
- `--output` - Output directory for PNG files (default: data/output/png)
- `--dpi` - Resolution for image conversion (default: 300)

**Output:**
- PNG images for each page
- Conversion info file

---

### **2. `02_detect_table_grid.py` - Table Grid Detection**
**What it does:** Detects table grids in PNG images using OpenCV morphological operations.

**Usage:**
```bash
python3 scripts/02_detect_table_grid.py page_001.png --output data/output/grid_detection
```

**Arguments:**
- `page_001.png` - Path to the PNG image (required)
- `--output` - Output directory for results (default: data/output/grid_detection)

**Output:**
- Prepared image for line detection
- Individual section images
- Grid detection results for each section
- Combined grid image
- Detection info file

---

### **3. `03_extract_text_from_cells.py` - Cell Identification and Text Extraction**
**What it does:** Identifies table cells and extracts text using Tesseract OCR.

**Usage:**
```bash
python3 scripts/03_extract_text_from_cells.py 04_combined_grid.png page_001.png --output data/output/text_extraction
```

**Arguments:**
- `04_combined_grid.png` - Path to the grid image (required)
- `page_001.png` - Path to the original PNG image (required)
- `--output` - Output directory for results (default: data/output/text_extraction)

**Output:**
- Cell visualization with bounding boxes
- Individual cell images
- Binary cell images for OCR
- Extracted text data (JSON)
- CSV format of extracted text

---

### **4. `04_create_excel_table.py` - Table Reconstruction and Excel Export**
**What it does:** Reconstructs table structure and exports to Excel with formatting.

**Usage:**
```bash
python3 scripts/04_create_excel_table.py extracted_text.json --output data/output/excel --filename my_table.xlsx
```

**Arguments:**
- `extracted_text.json` - Path to the extracted text JSON file (required)
- `--output` - Output directory for Excel file (default: data/output/excel)
- `--filename` - Excel filename (default: auto-generated)

**Output:**
- Excel file with professional formatting
- Table summary (JSON)
- Summary report (text)

---

### **5. `run_complete_pipeline.py` - Master Script**
**What it does:** Runs all steps in sequence automatically.

**Usage:**
```bash
python3 scripts/run_complete_pipeline.py input.pdf --output data/output --verbose
```

**Arguments:**
- `input.pdf` - Path to the input PDF file (required)
- `--output` - Base output directory (default: data/output)
- `--verbose` - Enable verbose logging

**Output:**
- All outputs from individual steps
- Complete pipeline summary
- Execution time

## 🚀 **Running Individual Steps**

### **For Debugging or Custom Processing:**
```bash
# Step 1: Convert PDF to PNG
python3 scripts/01_pdf_to_png.py data/input/your_file.pdf

# Step 2: Detect table grid
python3 scripts/02_detect_table_grid.py data/output/png/page_001.png

# Step 3: Extract text from cells
python3 scripts/03_extract_text_from_cells.py data/output/grid_detection/04_combined_grid.png data/output/png/page_001.png

# Step 4: Create Excel table
python3 scripts/04_create_excel_table.py data/output/text_extraction/extracted_text.json
```

### **For Complete Pipeline:**
```bash
# Run everything at once
python3 scripts/run_complete_pipeline.py data/input/your_file.pdf
```

## 🔍 **Debugging and Analysis**

Each script generates detailed output files that help you understand what's happening:

- **Debug Images:** See the image processing steps
- **JSON Data:** Structured data for analysis
- **Log Files:** Detailed processing information
- **CSV Files:** Easy-to-view text data

## 📁 **Output Directory Structure**

```
data/output/
├── png/                    # Step 1: PNG images
├── grid_detection/         # Step 2: Grid detection results
├── text_extraction/        # Step 3: Text extraction results
└── excel/                  # Step 4: Final Excel files
```

## 💡 **Tips for Best Results**

1. **High DPI:** Use 300+ DPI for better OCR accuracy
2. **Clear Tables:** Ensure tables have visible grid lines
3. **Good Contrast:** High contrast between text and background
4. **Individual Steps:** Run steps individually to debug issues
5. **Check Outputs:** Review intermediate files to understand the process

## 🚨 **Troubleshooting**

- **No Grid Detected:** Check if table has clear lines, adjust kernel sizes
- **Poor OCR:** Increase DPI, check image quality
- **Missing Cells:** Reduce minimum area threshold
- **Wrong Order:** Check row/column grouping parameters

## 🔧 **Customization**

Each script can be modified independently:
- Adjust morphological operation parameters
- Change OCR settings
- Modify cell detection thresholds
- Customize Excel formatting

This modular approach makes it easy to experiment and improve each step!
