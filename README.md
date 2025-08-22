# 📄 Document Parser: PDF to Excel Extractor

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A Python-based web application that intelligently extracts unstructured tabular data from PDF documents and converts it into a clean, structured Excel spreadsheet. This project is specifically tailored to parse complex employee punch-in/out records from a WMS report format.

---

## ## 🚀 Overview

Many organizations generate reports as PDFs that contain valuable data in a format that is difficult to use for analysis. This tool solves that problem by providing a simple web interface to upload a PDF file, automatically parsing the content, and making the data available for download as a user-friendly Excel file.

The core of this project is a robust parsing engine that can handle multi-line records and messy data layouts, which traditional table extraction tools often fail to process correctly.

---

## ## ✨ Features

* **Simple Web Interface:** Built with Streamlit for an intuitive user experience.
* **PDF Upload:** Easily upload multi-page PDF files directly from your browser.
* **Intelligent Parsing:** The backend logic is designed to understand the specific, complex structure of the WMS punch record report.
* **Data Structuring:** Converts the extracted raw text into a clean, labeled table format.
* **Excel Export:** Download the structured data as an `.xlsx` file with a single click.

---

## ##  Demo

Here is a quick look at the application's user interface.



---

## ## ⚙️ Tech Stack

* **Backend:** Python
* **Web Framework:** Streamlit
* **PDF Processing:** PyMuPDF (`fitz`)
* **Data Manipulation:** Pandas
* **Excel Engine:** openpyxl

---

## ## 📂 Project Structure

The project is organized to separate the user interface from the core parsing logic, making it clean and scalable.

```
document-parser-project/
├── 📂 data/
│   ├── 📂 input/
│   └── 📂 output/
├── 📂 src/
│   ├── __init__.py
│   └── 📄 parser.py
├── 📄 app.py
├── 📄 requirements.txt
└── 📄 README.md
```

---

## ## 🛠️ Setup and Installation

Follow these steps to set up the project on your local machine.

### ### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/document-parser-project.git](https://github.com/your-username/document-parser-project.git)
cd document-parser-project
```

### ### 2. Create a Virtual Environment (Recommended)

It's a best practice to create a virtual environment to manage project dependencies.

* **Windows:**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
* **macOS / Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### ### 3. Install Python Dependencies

Install all the required libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

---

## ## ▶️ How to Run the Application

Once the setup is complete, you can run the Streamlit application with a single command.

1.  Make sure you are in the root directory of the project (`document-parser-project/`).
2.  Run the following command in your terminal:

    ```bash
    streamlit run app.py
    ```

3.  Your web browser should automatically open a new tab with the application running. If not, open your browser and go to `http://localhost:8501`.

---

## ## 📖 How to Use

1.  **Launch the App:** Run the command above to start the local server.
2.  **Upload a File:** Click the "Choose a PDF file" button and select the WMS punch record PDF.
3.  **Parse the Document:** After uploading, click the "🚀 Parse File and Generate Excel" button.
4.  **Download:** A preview of the extracted data will appear. Click the "📥 Download Excel File" button to save the result to your computer.

---

## ## 💡 Future Improvements

This project has a solid foundation, but here are some potential enhancements:

* **Support for More File Types:** Add functionality to parse text from image files (`.png`, `.jpg`) using an OCR engine like Tesseract.
* **Advanced Data Cleaning:** Implement more sophisticated rules to handle edge cases and potential OCR errors.
* **Support for Different Templates:** Allow users to define parsing rules for different types of PDF layouts.
* **Database Integration:** Add an option to save the extracted data directly to a database (like SQLite or PostgreSQL).
* **Cloud Deployment:** Deploy the application to a cloud service like Streamlit Community Cloud or Heroku to make it publicly accessible.

---

## ## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## ## 📧 Contact

**Debabrata Ghosh** - [LinkedIn](https://www.linkedin.com/in/debabrataghoosh/) - [GitHub](https://github.com/debabrataghoosh)

Project Link: [https://github.com/debabrataghoosh/document-parser-project](https://github.com/debabrataghoosh/document-parser-project)
