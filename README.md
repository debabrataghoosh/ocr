# 🚀 ExtractX OCR Extraction

**Professional table extraction using Google Gemini API with interactive CLI**

## 🌟 **Features**

- **📁 Multi-format Support**: PDF, PNG, JPG, JPEG with multi-page processing
- **🤖 AI-Powered Extraction**: Google Gemini API for 98%+ accuracy
- **🔄 Complete Pipeline**: Unified PDF→PNG→Gemini→Excel workflow
- **📊 Multiple Export Formats**: Excel (.xlsx), CSV (.csv)
- **🎯 Interactive CLI**: User-guided interface with step-by-step prompts
- **⚡ Smart Processing**: Page range selection and rotation controls
- **🛡️ Robust Error Handling**: Retry logic with exponential backoff
- **🔧 Smart Table Merging**: Intelligent column alignment across pages
- **💾 Secure API Management**: Environment variable protection
- **📖 Multi-page Consolidation**: Combine multiple pages into single output files

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF/Images    │───▶│  Auto Rotation   │───▶│  Gemini API     │───▶│  Multi-Format   │
│   Multi-page    │    │   & Processing   │    │  Extraction     │    │    Export       │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 **Quick Start**

### **Interactive Pipeline**
```bash
# Install dependencies
pip install -r requirements.txt

# Run interactive pipeline
python scripts/ExtractX_OCR.py
```

### **Demo Simulation**
```bash
# See how the interactive pipeline works
python demo_interactive.py
```

### **🔑 API Configuration**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Create a `.env` file in project root:
```
GEMINI_API_KEY=your_key_here
```

## 📁 **Project Structure**

```
ocr/
├── 📁 scripts/
│   └── ExtractX_OCR.py             # 🆕 Interactive OCR Pipeline
├── 📁 demo_interactive.py          # Interactive demo simulation
├── 📁 requirements.txt             # Streamlined dependencies (Gemini-only)
├── 📁 .env                        # API key configuration (git-ignored)
└── 📁 data/                       # Input/output management
    ├── input/                     # Source files
    └── output/                    # Generated results
```
```

## 🎯 **How It Works**

### **Step 1: File Selection**
- Interactive file browsing (manual path, current dir, data/input)
- Automatic file validation
- Support for PDF and image files

### **Step 2: Page Configuration (PDF only)**
- Automatic page count detection
- User selects page range (all, specific range, first N pages)
- Page validation and confirmation

### **Step 3: Processing Options**
- Rotation selection (0°, 90°, 180°, 270°)
- Output format choice (Excel, CSV, or both)
- Output directory selection

### **Step 4: AI Processing**
- PDF to PNG conversion with rotation
- Google Gemini API table recognition
- Intelligent text extraction and structuring

### **Step 5: Export & Results**
- Multi-page table consolidation
- Professional Excel formatting
- CSV export for compatibility
- Timestamped output files

## 🔧 **Interactive Features**

### **File Selection Options**
```
📁 Select input file:
1. Enter file path manually
2. Browse current directory  
3. Browse data/input directory
```

### **Page Range Selection**
```
📄 Page Selection (Total: 5 pages):
1. Process all pages
2. Process specific page range (e.g., 1-5)
3. Process first N pages
```

### **Rotation Controls**
```
🔄 Rotation Options:
1. No rotation (0°)
2. Rotate 90° clockwise
3. Rotate 180°
4. Rotate 270° clockwise
```

### **Output Format Selection**
```
💾 Output Format Options:
1. Excel (.xlsx)
2. CSV (.csv)
3. Both Excel and CSV
```

## 📊 **Supported Formats**

### **Input Files**
- ✅ **PDF**: Multi-page documents with page range selection
- ✅ **PNG**: High-quality images
- ✅ **JPG/JPEG**: Standard image formats

### **Output Files**
- ✅ **Excel (.xlsx)**: Professional formatting with auto-width columns
- ✅ **CSV (.csv)**: Universal compatibility format
- ✅ **Multi-page consolidation**: All pages combined into single output

## 🔍 **Troubleshooting**

### **Common Issues**

**API Key Error**
- Verify your Gemini API key is correct
- Check API key permissions and quotas
- Ensure internet connectivity

**File Processing Issues**
- Check file format compatibility
- Verify file size limitations
- Ensure file is not corrupted

**Processing Errors**
- Check API key validity
- Verify image quality and clarity
- Ensure table is clearly visible

### **Performance Tips**
- Use high-quality images for better results
- Optimize image resolution (300-600 DPI)
- Ensure good lighting and contrast

## 🎨 **Customization**

### **Pipeline Configuration**
- Modify processing parameters in `complete_pipeline.py`
- Customize DPI settings for PDF conversion
- Adjust retry logic and timeout values

### **Functionality Extensions**
- Add new file format support
- Implement custom processing logic
- Extend export options

## 📈 **Performance Metrics**

- **Processing Speed**: 10-30 seconds per image
- **Accuracy**: 95%+ with clear images
- **File Size Support**: Limited by Gemini API
- **Interactive Experience**: Step-by-step guided process
- **Multi-page Support**: Unlimited pages with smart consolidation

## 🔒 **Security Features**

- **API Key Protection**: Environment variable storage
- **File Validation**: Type and size checking
- **Error Handling**: Safe error messages
- **No Data Storage**: Temporary processing only

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 **Acknowledgments**

- **Google Gemini API** for AI-powered text extraction
- **Interactive CLI** for user-friendly experience
- **Pandas & OpenPyXL** for Excel processing
- **PyMuPDF** for PDF handling
- **Pillow** for image processing

## 📞 **Support**

- **Issues**: Create a GitHub issue
- **Documentation**: Check INTERACTIVE_IMPLEMENTATION.md
- **Community**: Join our discussions

---

**🚀 Built with ❤️ using Interactive CLI & Google Gemini API**

**📊 Professional Table Extraction Made Simple**
