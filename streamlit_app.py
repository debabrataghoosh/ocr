#!/usr/bin/env python3
"""ExtractX OCR Extraction
High-accuracy (LLM-assisted) OCR -> Table -> Excel/CSV pipeline.

Pipeline Stages:
 1. File type detection
 2. Orientation auto-detect & correction + manual override preview
 3. Grid / cell detection preview
 4. OCR + LLM table structuring
 5. Export (Excel / CSV)

Supports: PDF (multi-page), JPEG, PNG. Batch upload.
"""

import os
import io
import json
import base64
import tempfile
from datetime import datetime
from dataclasses import dataclass
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
try:
    import cv2  # optional
except Exception:  # pragma: no cover
    cv2 = None
from dotenv import load_dotenv

# Optional OCR engines (loaded lazily)
try:
    import easyocr  # type: ignore  # will be unused in Gemini-only mode
except Exception:  # pragma: no cover
    easyocr = None
try:
    import pytesseract  # type: ignore  # will be unused in Gemini-only mode
except Exception:  # pragma: no cover
    pytesseract = None

try:
    import fitz  # PyMuPDF
except Exception as e:  # pragma: no cover
    fitz = None

SUPPORTED_FORMATS = ["pdf", "jpeg", "jpg", "png"]
STAGES = [
    (20, "File type detection ✅"),
    (40, "Orientation check/correction 🔄"),
    (60, "Preparing table extraction �"),
    (90, "Gemini table extraction 🎯"),
    (100, "Done ✅")
]

load_dotenv()  # Load environment variables from local .env if present

st.set_page_config(page_title="ExtractX OCR Extraction", page_icon="📊", layout="wide")
st.title("ExtractX OCR Extraction")
st.caption("LLM-assisted high-accuracy OCR to structured Excel/CSV")

# Sidebar info
def load_gemini_key_from_env_text(env_text: str) -> str | None:
    for line in env_text.splitlines():
        if line.strip().startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        if k.strip() == 'GEMINI_API_KEY':
            return v.strip().strip('"').strip("'")
    return None

with st.sidebar:
    st.markdown("### Pipeline Progress")
    st.markdown("1. Detect file type\n2. Orientation fix\n3. Grid & cells\n4. OCR + LLM\n5. Export")
    st.markdown("### Supported Formats")
    st.info("PDF (multi-page), JPEG, PNG. Batch upload supported.")
    st.markdown("### LLM (Gemini) Integration")
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = os.getenv('GEMINI_API_KEY', '')
    uploaded_env = st.file_uploader("Upload .env (optional)", type=['env'])
    if uploaded_env:
        env_bytes = uploaded_env.read().decode('utf-8', errors='ignore')
        parsed_key = load_gemini_key_from_env_text(env_bytes)
        if parsed_key:
            st.session_state.gemini_api_key = parsed_key
            st.success("Gemini key loaded from uploaded .env")
        else:
            st.warning("No GEMINI_API_KEY found in uploaded file")
    manual_toggle = st.checkbox("Enter / override key manually")
    if manual_toggle:
        manual_key = st.text_input("GEMINI_API_KEY", type="password")
        if manual_key:
            st.session_state.gemini_api_key = manual_key
    gemini_api_key = st.session_state.gemini_api_key
    if gemini_api_key:
        st.success("Gemini key active")
    else:
        st.info("No Gemini key: running heuristic OCR only")
    use_llm = bool(gemini_api_key)
    st.markdown("### Batch Orientation")
    if 'apply_global_orientation' not in st.session_state:
        st.session_state.apply_global_orientation = True
    if 'global_orientation_angle' not in st.session_state:
        st.session_state.global_orientation_angle = 0
    st.session_state.apply_global_orientation = st.checkbox(
        "Apply one rotation to every page/image", value=st.session_state.apply_global_orientation
    )
    if st.session_state.apply_global_orientation:
        st.session_state.global_orientation_angle = st.select_slider(
            "Global Manual Rotation (°)", options=[0,90,180,270], value=st.session_state.global_orientation_angle
        )
    else:
        st.caption("Per-page manual sliders will be shown in processing panels.")
    st.markdown("### Accuracy Note")
    st.write("Gemini-first pipeline. Local OCR fallback disabled for slim install.")
    st.markdown("### View Mode")
    if 'show_page_details' not in st.session_state:
        st.session_state.show_page_details = False
    st.session_state.show_page_details = st.checkbox(
        "Show per-page details (images, grids, intermediate tables)",
        value=st.session_state.show_page_details,
        help="Disable to process silently and only show final combined result."
    )
    st.markdown("---")

# Data classes
@dataclass
class PageResult:
    file_name: str
    page_index: int
    original_image: Image.Image
    processed_image: Image.Image
    angle_auto: int
    angle_manual: int
    grid_overlay: Image.Image | None
    cells: list
    df: pd.DataFrame | None


def is_supported(filename: str) -> bool:
    return filename.split('.')[-1].lower() in SUPPORTED_FORMATS


def load_pdf(path: str) -> list[Image.Image]:
    if fitz is None:
        raise RuntimeError("PyMuPDF not installed")
    doc = fitz.open(path)
    pages = []
    for p in doc:
        pix = p.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(img)
    return pages


def load_file(uploaded_file) -> list[Image.Image]:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name
    ext = suffix.replace('.', '')
    if ext == 'pdf':
        return load_pdf(temp_path)
    else:
        return [Image.open(temp_path).convert("RGB")]


def auto_orientation_angle(pil_image: Image.Image) -> int:
    """Detect orientation angle (0/90/180/270) using Tesseract OSD if available; else heuristic."""
    if pytesseract is not None:
        try:
            osd = pytesseract.image_to_osd(pil_image)
            for line in osd.splitlines():
                if 'Rotate:' in line:
                    deg = int(line.split(':')[-1].strip())
                    return deg
        except Exception:
            pass
    # Heuristic: assume landscape if width>height else 0
    w, h = pil_image.size
    if h > w * 1.2:
        return 0  # assume correct portrait
    return 0


def rotate(pil_image: Image.Image, angle: int) -> Image.Image:
    if angle == 0:
        return pil_image
    return pil_image.rotate(-angle, expand=True)


def detect_grid_and_cells(pil_image: Image.Image) -> tuple[Image.Image, list]:
    """Optional grid detection; returns overlay & cell boxes. If OpenCV missing, returns original image and empty cells."""
    if cv2 is None:
        return pil_image, []
    try:
        img = np.array(pil_image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
        hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        h_temp = cv2.morphologyEx(thr, cv2.MORPH_OPEN, hori_kernel, iterations=1)
        v_temp = cv2.morphologyEx(thr, cv2.MORPH_OPEN, vert_kernel, iterations=1)
        grid = cv2.add(h_temp, v_temp)
        grid = cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = img.copy()
        cells = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < 300:
                continue
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cells.append((x, y, w, h))
        return Image.fromarray(overlay), cells
    except Exception:
        return pil_image, []


def ocr_cells(pil_image: Image.Image, cells: list) -> list[list[str]]:
    """Run OCR cell by cell returning list of rows (simple left-to-right top-to-bottom grouping)."""
    if not cells:
        # fallback full image
        text = run_ocr(pil_image)
        return [[line] for line in text.splitlines() if line.strip()]
    # Sort cells by y then x
    cells_sorted = sorted(cells, key=lambda b: (round(b[1] / 30), b[0]))
    rows = []
    current_y = None
    row = []
    for (x, y, w, h) in cells_sorted:
        cell_img = pil_image.crop((x, y, x + w, y + h))
        txt = run_ocr(cell_img).strip()
        if current_y is None:
            current_y = y
        if abs(y - current_y) > 25 and row:
            rows.append(row)
            row = []
            current_y = y
        row.append(txt)
    if row:
        rows.append(row)
    return rows


def run_ocr(pil_image: Image.Image) -> str:
    """Local OCR disabled in Gemini-first mode. Returns empty string."""
    return ""


def llm_table_refine(rows: list[list[str]], api_key: str | None) -> pd.DataFrame:
    """Deprecated path retained for compatibility; returns empty when Gemini mode active."""
    return pd.DataFrame(rows)


def gemini_table_extract(pil_image: Image.Image, api_key: str) -> pd.DataFrame:
    """Send whole image to Gemini for table extraction -> DataFrame."""
    try:
        buffered = io.BytesIO()
        pil_image.save(buffered, format='PNG')
        b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        prompt = (
            "Extract all tables. Return pure JSON: {\"headers\": [...], \"rows\": [[...],[...]]}. "
            "No commentary. If multiple tables, vertically concatenate."
        )
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/png", "data": b64}}
                ]
            }],
            "generationConfig": {"temperature": 0.05, "topK": 1, "topP": 1, "maxOutputTokens": 4096}
        }
        import requests
        r = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            params={"key": api_key},
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        parts = data.get('candidates', [{}])[0].get('content', {}).get('parts', [])
        if not parts:
            return pd.DataFrame()
        txt = parts[0].get('text', '')
        jstart = txt.find('{')
        jend = txt.rfind('}') + 1
        if jstart == -1 or jend == 0:
            return pd.DataFrame()
        js = json.loads(txt[jstart:jend])
        headers = js.get('headers') or []
        rows = js.get('rows') or []
        if headers and rows:
            return pd.DataFrame(rows, columns=headers)
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def export_bytes(df: pd.DataFrame, kind: str) -> tuple[bytes, str, str]:
    if kind == 'xlsx':
        bio = io.BytesIO()
        df.to_excel(bio, index=False)
        bio.seek(0)
        return bio.getvalue(), f"extractx_{datetime.now():%Y%m%d_%H%M%S}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        csv_bytes = df.to_csv(index=False).encode()
        return csv_bytes, f"extractx_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv"


def process_page(file_name: str, page_index: int, image: Image.Image, gemini_key: str | None, *, use_global_orientation: bool, global_angle: int, show_details: bool) -> PageResult:
    progress = None
    if show_details:
        progress = st.progress(0, text="Initializing page...")
        progress.progress(STAGES[0][0], text="Detecting file type...")
    # Orientation
    angle_auto = auto_orientation_angle(image)
    rotated_auto = rotate(image, angle_auto)
    if progress:
        progress.progress(STAGES[1][0], text="Correcting orientation...")
    # Apply global rotation first (if any)
    effective_global = global_angle if use_global_orientation else 0
    after_global = rotate(rotated_auto, effective_global)
    # Optional per-page fine rotation always available in detailed mode
    manual_key = f"page_manual_delta_{file_name}_{page_index}"
    manual_delta = 0
    if show_details:
        manual_delta = st.select_slider(
            f"Per-page fine rotation (°) – {file_name} p{page_index+1}",
            options=[-270,-180,-90,0,90,180,270], value=0, key=manual_key
        )
        if effective_global:
            st.caption(f"Global {effective_global}° + Page Δ {manual_delta}°")
        else:
            st.caption(f"Page Δ {manual_delta}° applied")
    final_img = rotate(after_global, manual_delta)
    grid_overlay = None
    cells = []
    if show_details:
        st.image(final_img, caption=f"Final Preview (auto {angle_auto}° / global {effective_global}° / page Δ {manual_delta}°)")
        grid_overlay, cells = detect_grid_and_cells(final_img)
        if cv2 is not None and cells:
            st.image(grid_overlay, caption="Grid / Cells Preview")
        progress and progress.progress(STAGES[2][0], text="Preparing image for Gemini...")
        if gemini_key:
            df = gemini_table_extract(final_img, gemini_key)
        else:
            df = pd.DataFrame()
        if not df.empty:
            st.dataframe(df.head(50), use_container_width=True)
        progress and progress.progress(STAGES[3][0], text="Gemini extraction...")
        progress and progress.progress(STAGES[4][0], text="Page complete")
    else:
        if gemini_key:
            df = gemini_table_extract(final_img, gemini_key)
        else:
            df = pd.DataFrame()
    total_manual = effective_global + manual_delta
    return PageResult(file_name, page_index, image, final_img, angle_auto, total_manual, grid_overlay, cells, df)


def main():  # noqa: D401
    st.markdown("### Upload Files")
    uploaded_files = st.file_uploader(
        "Upload PDF / JPEG / PNG", type=SUPPORTED_FORMATS, accept_multiple_files=True
    )
    if not uploaded_files:
        st.info("Upload one or more supported documents to begin.")
        return
    all_results: list[PageResult] = []
    total_pages = 0
    file_pages_cache = []  # (file, pages)
    for uf in uploaded_files:
        if not is_supported(uf.name):
            st.error(f"Unsupported file skipped: {uf.name}")
            continue
        pages = []
        try:
            pages = load_file(uf)
        except Exception as e:
            st.error(f"Failed to load {uf.name}: {e}")
            continue
        file_pages_cache.append((uf.name, pages))
        total_pages += len(pages)

    if not st.session_state.show_page_details:
        overall_bar = st.progress(0, text="Starting batch processing...")
        processed = 0
        for fname, pages in file_pages_cache:
            for i, pg in enumerate(pages):
                res = process_page(
                    fname,
                    i,
                    pg,
                    gemini_api_key if use_llm else None,
                    use_global_orientation=st.session_state.apply_global_orientation,
                    global_angle=st.session_state.global_orientation_angle,
                    show_details=False,
                )
                all_results.append(res)
                processed += 1
                pct = int((processed / max(total_pages,1)) * 100)
                overall_bar.progress(pct, text=f"Processing pages... {processed}/{total_pages}")
        overall_bar.progress(100, text="Batch complete")
    else:
        for fname, pages in file_pages_cache:
            st.subheader(f"File: {fname}")
            for i, pg in enumerate(pages):
                with st.expander(f"Page {i+1} • Details", expanded=True if len(pages)==1 else False):
                    res = process_page(
                        fname,
                        i,
                        pg,
                        gemini_api_key if use_llm else None,
                        use_global_orientation=st.session_state.apply_global_orientation,
                        global_angle=st.session_state.global_orientation_angle,
                        show_details=True,
                    )
                    all_results.append(res)

    # Aggregate export
    st.markdown("### Export Consolidated Results")
    valid_dfs = [r.df for r in all_results if r.df is not None and not r.df.empty]
    if not valid_dfs:
        st.warning("No extracted tables to export yet.")
        return
    combined = pd.concat(valid_dfs, ignore_index=True)
    st.write(f"Combined rows: {combined.shape[0]} | columns: {combined.shape[1]}")
    excel_bytes, excel_name, excel_mime = export_bytes(combined, 'xlsx')
    csv_bytes, csv_name, csv_mime = export_bytes(combined, 'csv')
    json_bytes = combined.to_json(orient='records', force_ascii=False, indent=2).encode('utf-8')
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Download Excel", data=excel_bytes, file_name=excel_name, mime=excel_mime, use_container_width=True)
    with c2:
        st.download_button("Download CSV", data=csv_bytes, file_name=csv_name, mime=csv_mime, use_container_width=True)
    with c3:
        st.download_button("Download JSON", data=json_bytes, file_name=f"extractx_{datetime.now():%Y%m%d_%H%M%S}.json", mime="application/json", use_container_width=True)

    st.caption("Processing complete. Full consolidated table available for download.")


if __name__ == '__main__':
    main()
