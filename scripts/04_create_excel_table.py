#!/usr/bin/env python3
"""
Script 4: Table Reconstruction and Excel Export
Reconstructs table structure and exports to Excel with formatting.
"""

import os
import sys
import logging
import json
import argparse
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Border, Side, Font, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_extracted_text(text_data_path: str):
    """
    Load extracted text data from JSON file.
    
    Args:
        text_data_path: Path to the extracted text JSON file
        
    Returns:
        Dictionary containing table data and cell details
    """
    try:
        logger.info(f"📂 Loading extracted text data from: {os.path.basename(text_data_path)}")
        
        with open(text_data_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"   ✅ Loaded data: {data['total_rows']} rows, {data['total_cells']} cells")
        return data
        
    except Exception as e:
        logger.error(f"❌ Error loading extracted text data: {e}")
        raise

def clean_table_data(table_data: list):
    """
    Clean and format the extracted table data.
    
    Args:
        table_data: List of lists containing table data
        
    Returns:
        Cleaned table data
    """
    logger.info("🧹 Cleaning and formatting table data...")
    
    if not table_data:
        return []
    
    # Find the maximum number of columns
    max_cols = max(len(row) for row in table_data)
    
    # Pad rows with fewer columns
    cleaned_data = []
    for row in table_data:
        # Clean cell values
        cleaned_row = []
        for cell in row:
            if isinstance(cell, str):
                cleaned_cell = cell.strip()
                # Replace common OCR artifacts
                cleaned_cell = cleaned_cell.replace('|', 'I')  # Common OCR mistake
                cleaned_cell = cleaned_cell.replace('0', 'O')  # Common OCR mistake
                cleaned_cell = cleaned_cell.replace('1', 'l')  # Common OCR mistake
            else:
                cleaned_cell = str(cell).strip()
            
            cleaned_row.append(cleaned_cell)
        
        # Pad row to match maximum columns
        while len(cleaned_row) < max_cols:
            cleaned_row.append("")
        
        cleaned_data.append(cleaned_row)
    
    logger.info(f"   ✅ Cleaned {len(cleaned_data)} rows with {max_cols} columns")
    return cleaned_data

def create_dataframe(table_data: list):
    """
    Create a pandas DataFrame from the cleaned table data.
    
    Args:
        table_data: Cleaned table data
        
    Returns:
        pandas DataFrame
    """
    logger.info("📊 Creating pandas DataFrame...")
    
    if not table_data:
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Remove completely empty rows and columns
    df = df.dropna(how='all').dropna(axis=1, how='all')
    
    logger.info(f"   ✅ DataFrame created: {df.shape[0]} rows × {df.shape[1]} columns")
    return df

def export_to_excel_with_formatting(df: pd.DataFrame, output_path: str):
    """
    Export DataFrame to Excel with professional formatting.
    
    Args:
        df: pandas DataFrame
        output_path: Path for the Excel file
    """
    try:
        logger.info(f"💾 Exporting to Excel: {os.path.basename(output_path)}")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create a new workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Extracted_Table"
        
        # Write data to worksheet
        for r in dataframe_to_rows(df, index=False, header=False):
            ws.append(r)
        
        # Auto-adjust column widths
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column_letter].width = adjusted_width
        
        # Add borders and formatting
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        # Apply borders and center alignment to all cells
        for row in ws.iter_rows():
            for cell in row:
                cell.border = thin_border
                cell.alignment = Alignment(horizontal='center', vertical='center')
        
        # Save the workbook
        wb.save(output_path)
        
        logger.info(f"   ✅ Excel file saved: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Error exporting to Excel: {e}")
        raise

def create_table_summary(df: pd.DataFrame, cell_details: list, output_dir: str):
    """
    Create a summary of the extracted table.
    
    Args:
        df: pandas DataFrame
        cell_details: List of cell details from extraction
        output_dir: Output directory
    """
    try:
        logger.info("📋 Creating table summary...")
        
        # Calculate statistics
        total_cells = df.shape[0] * df.shape[1]
        non_empty_cells = df.notna().sum().sum()
        empty_cells = total_cells - non_empty_cells
        
        # Create summary
        summary = {
            'table_dimensions': f"{df.shape[0]} rows × {df.shape[1]} columns",
            'total_cells': int(total_cells),  # Convert to regular int
            'non_empty_cells': int(non_empty_cells),  # Convert to regular int
            'empty_cells': int(empty_cells),  # Convert to regular int
            'fill_rate': f"{(non_empty_cells/total_cells)*100:.1f}%",
            'sample_data': df.head(5).to_dict('records') if not df.empty else []
        }
        
        # Save summary
        summary_path = os.path.join(output_dir, "table_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create summary report
        report_path = os.path.join(output_dir, "table_summary_report.txt")
        with open(report_path, 'w') as f:
            f.write("TABLE EXTRACTION SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dimensions: {summary['table_dimensions']}\n")
            f.write(f"Total Cells: {summary['total_cells']}\n")
            f.write(f"Non-Empty Cells: {summary['non_empty_cells']}\n")
            f.write(f"Empty Cells: {summary['empty_cells']}\n")
            f.write(f"Fill Rate: {summary['fill_rate']}\n\n")
            
            f.write("SAMPLE DATA (First 5 rows):\n")
            f.write("-" * 30 + "\n")
            for i, row in enumerate(summary['sample_data']):
                f.write(f"Row {i+1}: {list(row.values())}\n")
        
        logger.info(f"   💾 Summary saved: {summary_path}")
        logger.info(f"   💾 Report saved: {report_path}")
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Error creating table summary: {e}")
        raise

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Create Excel table from extracted text data')
    parser.add_argument('text_data', help='Path to the extracted text JSON file')
    parser.add_argument('--output', '-o', default='data/output/excel', 
                       help='Output directory for Excel file (default: data/output/excel)')
    parser.add_argument('--filename', '-f', default=None,
                       help='Excel filename (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.text_data):
        logger.error(f"❌ Text data file not found: {args.text_data}")
        sys.exit(1)
    
    try:
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        # Load extracted text data
        data = load_extracted_text(args.text_data)
        
        # Clean table data
        cleaned_data = clean_table_data(data['table_data'])
        
        # Create DataFrame
        df = create_dataframe(cleaned_data)
        
        if df.empty:
            logger.error("❌ No valid table data to export")
            sys.exit(1)
        
        # Generate filename if not provided
        if args.filename is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            args.filename = f"extracted_table_{timestamp}.xlsx"
        
        # Export to Excel
        excel_path = os.path.join(args.output, args.filename)
        export_to_excel_with_formatting(df, excel_path)
        
        # Create table summary
        summary = create_table_summary(df, data.get('cell_details', []), args.output)
        
        print(f"\n🎯 Table Creation Summary:")
        print(f"   Input Data: {args.text_data}")
        print(f"   Table Dimensions: {summary['table_dimensions']}")
        print(f"   Total Cells: {summary['total_cells']}")
        print(f"   Non-Empty Cells: {summary['non_empty_cells']}")
        print(f"   Fill Rate: {summary['fill_rate']}")
        print(f"   Excel File: {excel_path}")
        print(f"   Output Directory: {args.output}")
        
        # Show sample data
        if summary['sample_data']:
            print(f"\n📊 Sample Data (First 3 rows):")
            for i, row in enumerate(summary['sample_data'][:3]):
                print(f"   Row {i+1}: {list(row.values())}")
        
    except Exception as e:
        logger.error(f"❌ Table creation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
