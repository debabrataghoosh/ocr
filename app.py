#!/usr/bin/env python3
"""
PDF Table Extractor with Advanced Image Processing
Streamlit Web Application
"""

import streamlit as st
import os
import pandas as pd
from src.pdf_table_extractor import PDFTableExtractor

# Page configuration
st.set_page_config(
    page_title="PDF Table Extractor - Advanced Image Processing",
    page_icon="📊",
    layout="wide"
)

def main():
    """Main application function."""
    st.title("📊 PDF Table Extractor")
    st.markdown("**Extract tables from PDF using advanced image processing and line detection**")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF file to extract tables using advanced image processing"
    )
    
    # Page selection
    pages_input = st.sidebar.text_input(
        "Pages to process",
        value="all",
        help="Enter page numbers (e.g., '1,2,3') or 'all' for all pages"
    )
    
    # Processing options
    st.sidebar.header("🔧 Processing Options")
    show_sample = st.sidebar.checkbox("Show sample data", value=True)
    auto_format = st.sidebar.checkbox("Auto-format Excel", value=True)
    
    # Main content area
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = f"data/input/{uploaded_file.name}"
        os.makedirs("data/input", exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Process button
        if st.button("🚀 Extract Tables with Advanced Processing", type="primary"):
            with st.spinner("Processing PDF with advanced image processing..."):
                try:
                    # Initialize PDF table extractor
                    extractor = PDFTableExtractor()
                    
                    # Extract tables
                    tables = extractor.extract_tables_from_pdf(temp_path, pages_input)
                    
                    if tables:
                        st.success(f"🎉 Successfully extracted {len(tables)} tables!")
                        
                        # Display table summary
                        st.header("📊 Extraction Results")
                        summary = extractor.get_table_summary()
                        
                        for table_info in summary['tables']:
                            with st.expander(f"Table {table_info['table_number']}: {table_info['dimensions']}"):
                                st.write(f"**Total cells:** {table_info['total_cells']}")
                                st.write(f"**Non-empty cells:** {table_info['non_empty_cells']}")
                                
                                if show_sample and table_info['sample_data']:
                                    st.write("**Sample data (first 3 rows):**")
                                    sample_df = pd.DataFrame(table_info['sample_data'])
                                    st.dataframe(sample_df, use_container_width=True)
                        
                        # Export to Excel
                        st.header("💾 Export to Excel")
                        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                        excel_filename = f"advanced_extraction_tables_{timestamp}.xlsx"
                        excel_path = os.path.join("data/output/exel", excel_filename)
                        
                        excel_path = extractor.export_to_excel(excel_path)
                        
                        # Download button
                        with open(excel_path, "rb") as f:
                            excel_data = f.read()
                        
                        st.download_button(
                            label="📥 Download Excel with Table Structure",
                            data=excel_data,
                            file_name=excel_filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        st.info(f"💡 **Excel file saved:** {excel_path}")
                        st.info("✅ **Table structure preserved** - Grid detection and cell alignment maintained!")
                        
                    else:
                        st.error("❌ No tables could be extracted from the PDF")
                        st.info("💡 This might be a document without clear table structure.")
                
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {e}")
                    st.info("💡 Make sure the PDF contains tables and is not corrupted.")
    
    else:
        # Welcome message and instructions
        st.header("🎯 Welcome to PDF Table Extractor")
        st.markdown("""
        This application extracts tables from PDF files using **advanced image processing techniques** and **individual cell OCR**.
        
        ### ✨ Key Features:
        - **Advanced Image Processing** - OpenCV-based grid line detection
        - **Individual Cell OCR** - Tesseract on each cell for maximum accuracy
        - **Grid-Based Extraction** - Precise cell boundaries and alignment
        - **Professional Excel Output** - Maintains table structure and formatting
        
        ### 🚀 How It Works:
        1. **Convert PDF to images** with high resolution
        2. **Detect table grid lines** using OpenCV morphological operations
        3. **Identify every cell** with precise coordinates using contours
        4. **Perform OCR on individual cells** for maximum text accuracy
        5. **Rebuild table structure** and export to Excel
        
        ### 📋 Supported PDF Types:
        - **Scanned PDFs** with table structures (excellent results)
        - **Digital PDFs** with embedded tables
        - **Complex tables** with borders and merged cells
        - **Multi-page documents** with multiple tables
        
        ### 💡 Perfect For:
        - Scanned documents and printed tables
        - Financial reports and invoices
        - Data tables from printed materials
        - Any document where traditional PDF extraction fails
        """)
        
        # How it works expander
        with st.expander("🔍 How It Works - Technical Details"):
            st.markdown("""
            ### **Step 1: Image Preparation for Line Detection**
            - Convert PDF to high-quality images (300+ DPI)
            - Apply grayscale conversion and adaptive thresholding
            - Invert image (lines white, background black)
            - Use dilation to thicken and connect grid lines
            
            ### **Step 2: Grid Line Detection**
            - Isolate horizontal lines with morphological operations
            - Isolate vertical lines with morphological operations
            - Combine to form complete table grid
            - Clean up artifacts and noise
            
            ### **Step 3: Cell Identification**
            - Find cell contours using OpenCV's findContours
            - Sort contours top-to-bottom, left-to-right
            - Get precise bounding boxes for each cell
            - Filter by area to remove small artifacts
            
            ### **Step 4: Individual Cell OCR**
            - Crop each cell from original image
            - Run Tesseract on individual cells
            - Use PSM 7 (single line) for better accuracy
            - Clean and format extracted text
            
            ### **Step 5: Table Reconstruction**
            - Organize cell text by detected position
            - Create structured DataFrame
            - Export to Excel with proper formatting
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("**Powered by Advanced Image Processing Technology** | Grid Detection & Cell-by-Cell OCR")

if __name__ == "__main__":
    main()
