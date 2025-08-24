# 📊 PDF Table Extractor with Advanced Image Processing

**Extract tables from PDF files using computer vision and cell-by-cell OCR for maximum accuracy.**

## 🎯 **What This System Does**

This application extracts tables from PDF files using **advanced image processing techniques** and **individual cell OCR**. Unlike traditional PDF extraction methods, this system:

- ✅ **Detects table grids** using OpenCV morphological operations
- ✅ **Identifies every cell** with precise coordinates using contours
- ✅ **Performs OCR on individual cells** for maximum text accuracy
- ✅ **Maintains table structure** by rebuilding from detected grid
- ✅ **Works with scanned documents** and complex table layouts

## 🚀 **Key Features**

### **Advanced Image Processing**
- **Grid Line Detection**: Uses OpenCV to find horizontal and vertical table lines
- **Cell Identification**: Precise contour detection for every table cell
- **Image Preparation**: Adaptive thresholding and line thickening for better detection

### **Individual Cell OCR**
- **Cell-by-Cell Processing**: Runs Tesseract on each cell individually
- **Optimized PSM**: Uses PSM 7 (single line) for better accuracy
- **Text Cleaning**: Intelligent cleaning and formatting of extracted text

### **Table Reconstruction**
- **Position-Based Sorting**: Maintains correct row/column order
- **Structure Preservation**: Rebuilds table with exact cell alignment
- **Excel Export**: Professional formatting with borders and auto-sizing

## 📋 **Perfect For**

- **Scanned PDF documents** with table structures
- **Complex tables** with merged cells and borders
- **Financial reports** and invoices
- **Data tables** from printed documents
- **Any document** where traditional PDF extraction fails

## 🛠️ **Installation**

### **Prerequisites**
- Python 3.8+
- Tesseract OCR (required for text extraction)
- OpenCV (for image processing)

### **Install Dependencies**
```bash
# Clone the repository
git clone <repository-url>
cd ocr

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### **Install Tesseract OCR (Required)**
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

## 🚀 **Usage**

### **Web Interface (Recommended)**
```bash
streamlit run app.py
```
- Upload PDF files through the web interface
- Automatic table detection and extraction
- Download Excel files with proper table structure

### **Command Line**
```bash
python demo_pdf_table_extractor.py
```
- Process PDFs from command line
- View extraction results and statistics
- Export to Excel with formatting

## 🔍 **How It Works**

### **Step 1: Image Preparation for Line Detection**
1. **Load Image and Rotate**: Convert PDF to high-quality images
2. **Convert to Grayscale**: Single-channel image for line detection
3. **Apply Adaptive Thresholding**: Clean black-and-white image
4. **Invert the Image**: Make lines white, background black
5. **Thicken the Lines**: Use dilation to connect small breaks

### **Step 2: Detect Horizontal and Vertical Grid Lines**
1. **Isolate Horizontal Lines**: Morphological operations with horizontal kernel
2. **Isolate Vertical Lines**: Morphological operations with vertical kernel
3. **Combine Lines**: Form complete table grid

### **Step 3: Identify Every Cell in the Grid**
1. **Find Cell Contours**: Use OpenCV's findContours function
2. **Sort the Contours**: Top-to-bottom, left-to-right ordering
3. **Get Bounding Boxes**: Precise coordinates for each cell

### **Step 4: Perform OCR on Each Cell Individually**
1. **Loop Through Sorted Cells**: Process each cell by position
2. **Crop the Cell**: Extract small cell area from original image
3. **Run Tesseract**: Use PSM 7 for single line of text
4. **Store the Text**: Organize results by table structure

### **Step 5: Rebuild the Table and Save**
1. **Create DataFrame**: Organize cell text by position
2. **Apply Final Cleaning**: Clean and format extracted data
3. **Export to Excel**: Maintain table structure and formatting

## 📁 **Project Structure**

```
ocr/
├── app.py                          # Main Streamlit application
├── demo_pdf_table_extractor.py     # Command-line demo
├── src/
│   └── pdf_table_extractor.py     # Core extraction engine
├── data/
│   ├── input/                      # PDF input files
│   └── output/                     # Excel output files
│       ├── exel/                   # Excel files
│       └── png/                    # Extracted images
├── requirements.txt                # Python dependencies
└── README.md                      # This file
```

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Set custom Tesseract path
export TESSERACT_PATH=/usr/local/bin/tesseract
```

### **Processing Options**
- **Page Selection**: Process specific pages or all pages
- **Image Quality**: Adjust DPI for better OCR accuracy
- **Line Detection**: Fine-tune morphological operation parameters

## 📊 **Output Format**

### **Excel Structure**
- **Multiple Sheets**: One sheet per table
- **Maintained Structure**: Same dimensions and layout as detected grid
- **Professional Formatting**: Borders, cell alignment, and spacing
- **Auto-sized Columns**: Optimal width for content readability

### **Data Quality**
- **Grid-Based Extraction**: Accurate cell boundaries and alignment
- **Individual Cell OCR**: Maximum text accuracy per cell
- **Structure Preservation**: Maintains detected table layout

## 🚨 **Troubleshooting**

### **Common Issues**

#### **No Table Grid Detected**
- PDF might not have clear table lines
- Try adjusting image preprocessing parameters
- Check if document has actual table structure

#### **Poor OCR Accuracy**
- Ensure high-quality image conversion (300+ DPI)
- Check Tesseract installation and configuration
- Verify image preprocessing quality

#### **Incorrect Cell Detection**
- Adjust morphological operation parameters
- Check for complex table layouts
- Verify line thickness and contrast

### **Performance Tips**
- **High DPI**: Use 300+ DPI for better OCR accuracy
- **Page Selection**: Process specific pages for faster results
- **Image Quality**: Ensure good contrast in original document

## 🔮 **Advanced Features**

### **Custom Grid Detection**
- Modify morphological operation parameters
- Add custom line detection algorithms
- Implement industry-specific table recognition

### **Batch Processing**
- Process multiple PDFs automatically
- Batch Excel export with consistent formatting
- Integration with document management systems

### **API Integration**
- RESTful API for programmatic access
- Integration with web applications
- Automated document processing workflows

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 **Acknowledgments**

- **OpenCV**: Computer vision and image processing
- **Tesseract**: OCR capabilities for text extraction
- **pdf2image**: PDF to image conversion
- **Streamlit**: Web interface framework

---

**Built with ❤️ for advanced table extraction using computer vision**
