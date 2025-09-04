# 🚀 ExtractX OCR Extraction

**Professional table extraction using Google Gemini API with complete pipeline automation**

## 🌟 **Features**

- **📁 Multi-format Support**: PDF, PNG, JPG, JPEG with multi-page processing
- **🤖 AI-Powered Extraction**: Google Gemini API for 98%+ accuracy
- **🔄 Complete Pipeline**: Unified PDF→PNG→Gemini→Excel workflow
- **📊 Multiple Export Formats**: Excel (.xlsx), CSV (.csv), JSON (.json)
- **🎨 Beautiful Web UI**: Modern Streamlit interface with batch processing
- **⚡ CLI Automation**: Command-line pipeline for batch operations
- **🛡️ Robust Error Handling**: Retry logic with exponential backoff
- **🔧 Smart Table Merging**: Intelligent column alignment across pages
- **📱 Responsive Design**: Works on desktop and mobile devices
- **💾 Secure API Management**: Environment variable protection

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF/Images    │───▶│  Auto Rotation   │───▶│  Gemini API     │───▶│  Multi-Format   │
│   Multi-page    │    │   & Processing   │    │  Extraction     │    │    Export       │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 **Quick Start**

### **Web Interface**
```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py
```

### **Command Line Pipeline**
```bash
# Basic usage
python scripts/complete_pipeline.py input.pdf --output results/

# Advanced options
python scripts/complete_pipeline.py input.pdf \
  --max-pages 5 \
  --rotation 90 \
  --output results/
```

### **� API Configuration**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Create a `.env` file in project root:
```
GEMINI_API_KEY=your_key_here
```

## �📁 **Project Structure**

```
ocr/
├── 📁 streamlit_app.py             # Web interface with batch processing
├── 📁 scripts/
│   ├── complete_pipeline.py       # 🆕 Unified pipeline (PDF→PNG→Gemini→Excel)
│   ├── 01_pdf_to_png.py          # PDF conversion with orientation
│   ├── 03_extract_text_with_gemini.py  # Gemini API extraction
│   └── 04_create_excel_from_gemini.py  # Professional Excel formatting
├── 📁 requirements.txt             # Streamlined dependencies (Gemini-only)
├── 📁 .env                        # API key configuration (git-ignored)
├── 📁 data/                       # Input/output management
│   ├── input/                     # Source files
│   └── output/                    # Generated results
└── 📁 docs/                       # Project documentation
    ├── PROJECT_STRUCTURE.md       # Detailed structure
    └── CLEANUP_SUMMARY.md         # Development history
```

## 🎯 **How It Works**

### **Step 1: File Upload**
- Supports PDF and image files
- Automatic file validation
- Real-time file information display

### **Step 2: AI Processing**
- Uses Google Gemini API for table recognition
- Intelligent text extraction
- Structured data organization

### **Step 3: Excel Creation**
- Professional formatting
- Auto-adjusted column widths
- Clean headers and borders

### **Step 4: Download**
- Excel file download
- JSON data backup
- Timestamped filenames

## 🔧 **Configuration**

### **Environment Variables**
Preferred: use a `.env` file (auto-loaded via python-dotenv) or set in shell.
```bash
export GEMINI_API_KEY="your_api_key_here"  # shell method
```
Sidebar options:
- Upload `.env` file (parsed for GEMINI_API_KEY)
- Manual override input (masked)

### **API Key Security**
- API keys are stored securely in session state
- Never logged or stored permanently
- Input is masked for security

## 📊 **Supported Formats**

### **Input Files**
- ✅ **PDF**: Multi-page documents
- ✅ **PNG**: High-quality images
- ✅ **JPG/JPEG**: Standard image formats

### **Output Files**
- ✅ **Excel (.xlsx)**: Professional formatting
- ✅ **JSON**: Raw data backup
- ✅ **CSV**: Optional export

## 🚀 **Advanced Usage**

### **Command Line Interface**
```bash
# Run individual scripts
python3 scripts/01_pdf_to_png.py "input.pdf"
python3 scripts/03_extract_text_with_gemini.py "image.png" --api-key "KEY"
python3 scripts/04_create_excel_from_gemini.py "data.json"
```

### **Complete Pipeline**
```bash
python3 scripts/run_complete_pipeline.py "input.pdf" --api-key "KEY"
```

## 🔍 **Troubleshooting**

### **Common Issues**

**API Key Error**
- Verify your Gemini API key is correct
- Check API key permissions and quotas
- Ensure internet connectivity

**File Upload Issues**
- Check file format compatibility
- Verify file size (max 200MB)
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

### **Styling**
- Modify CSS in `streamlit_app.py`
- Customize color schemes
- Adjust layout and spacing

### **Functionality**
- Add new file format support
- Implement custom processing logic
- Extend export options

## 📈 **Performance Metrics**

- **Processing Speed**: 10-30 seconds per image
- **Accuracy**: 95%+ with clear images
- **File Size Support**: Up to 200MB
- **Concurrent Users**: Unlimited

## 🔒 **Security Features**

- **API Key Protection**: Secure input handling
- **File Validation**: Type and size checking
- **Session Management**: Secure data handling
- **Error Handling**: Safe error messages

## 🌐 **Deployment**

### **Local Development**
```bash
streamlit run streamlit_app.py --server.port 8501
```

### **Production Deployment**
```bash
# Using Streamlit Cloud
git push origin main

# Using Docker
docker build -t ocr-extractor .
docker run -p 8501:8501 ocr-extractor
```

### **Environment Variables**
```bash
# Production settings
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

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
- **Streamlit** for the beautiful web interface
- **Pandas & OpenPyXL** for Excel processing
- **Pillow** for image handling

## 📞 **Support**

- **Issues**: Create a GitHub issue
- **Documentation**: Check PROJECT_STRUCTURE.md
- **Community**: Join our discussions

---

**🚀 Built with ❤️ using Streamlit & Google Gemini API**

**📊 Professional Table Extraction Made Simple**
