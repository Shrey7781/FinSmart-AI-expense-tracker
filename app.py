from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import sys
import types
import traceback
from datetime import datetime, date, timedelta
import tempfile
import uuid
import shutil

# Matplotlib + FPDF for report generation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF

# ============================================================
# 🧠 Fix for 'TextCleaner' custom transformer (needed for joblib.load)
# ============================================================
text_cleaner_module = types.ModuleType("text_cleaner")

class TextCleaner:
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X

text_cleaner_module.TextCleaner = TextCleaner
sys.modules["text_cleaner"] = text_cleaner_module
sys.modules["__main__"].TextCleaner = TextCleaner

# ============================================================
# 🚀 FastAPI Configuration
# ============================================================
app = FastAPI(title="FinSmart AI API", version="2.7")

MODEL_PATH = "finsmart_expense_model.pkl"
DB_FILE = "FinSmart_DB.xlsx"

# Load ML model safely
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# ============================================================
# 📘 Pydantic Model for Manual Entry Input
# ============================================================
class TransactionInput(BaseModel):
    date: str
    amount: float
    description: str

# ============================================================
# 📘 Helper Functions
# ============================================================
def prepare_dataframe(df):
    # Ensure columns exist before operations
    if "date" not in df.columns:
        df["date"] = None
    if "amount" not in df.columns:
        df["amount"] = 0
    if "description" not in df.columns:
        df["description"] = ""

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").abs()
    df["description"] = df["description"].astype(str)
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["date"] = df["date"].dt.date
    return df

def serialize_dates(data):
    """Convert all date/datetime objects to strings for JSON serialization."""
    for row in data:
        if "date" in row and not isinstance(row["date"], str):
            row["date"] = str(row["date"])
    return data

def update_excel_database(new_data: pd.DataFrame):
    """Append new transactions and refresh the summary sheet."""
    try:
        if os.path.exists(DB_FILE):
            existing = pd.read_excel(DB_FILE, sheet_name="Transactions")
            df_all = pd.concat([existing, new_data], ignore_index=True)
        else:
            df_all = new_data

        # Ensure columns required for grouping
        if "predicted_category" not in df_all.columns:
            df_all["predicted_category"] = "Uncategorized"

        summary = (
            df_all.groupby("predicted_category")["amount"]
            .sum()
            .reset_index()
            .rename(columns={"predicted_category": "Category", "amount": "Total_Spent"})
            .sort_values(by="Total_Spent", ascending=False)
        )

        with pd.ExcelWriter(DB_FILE, engine="openpyxl") as writer:
            df_all.to_excel(writer, sheet_name="Transactions", index=False)
            summary.to_excel(writer, sheet_name="Summary", index=False)

        return True
    except Exception as e:
        print(f"⚠️ Warning: Failed to update Excel DB — {e}")
        return False

# ============================================================
# 🌐 API ROUTES
# ============================================================
@app.get("/")
def home():
    return {"message": "🚀 FinSmart AI Excel DB is active and ready!"}

# ------------------------------------------------------------
# 📤 Upload CSV File
# ------------------------------------------------------------
@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file with columns: date, amount, description
    Example: POST /upload-csv with form-data: file=@train.csv
    """
    try:
        if not model:
            return JSONResponse({"error": "Model not loaded properly."}, status_code=500)

        if not file.filename.endswith(".csv"):
            return JSONResponse({"error": "Please upload a valid CSV file."}, status_code=400)

        df = pd.read_csv(file.file)
        df = prepare_dataframe(df)

        # Check for required columns
        if not {"date", "amount", "description"}.issubset(df.columns):
            return JSONResponse({"error": "CSV must contain 'date', 'amount', 'description' columns."}, status_code=400)

        preds = model.predict(df[["description", "amount", "day", "month", "day_of_week"]])
        df["predicted_category"] = preds

        update_excel_database(df)

        entries = df.to_dict(orient="records")
        entries = serialize_dates(entries)

        summary = []
        if os.path.exists(DB_FILE):
            summary_df = pd.read_excel(DB_FILE, sheet_name="Summary")
            summary = summary_df.to_dict(orient="records")

        return JSONResponse({
            "message": "✅ CSV uploaded and processed successfully.",
            "records_added": len(df),
            "entries": entries,
            "summary": summary
        })

    except Exception as e:
        print("❌ Error in /upload-csv:", traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)

# ------------------------------------------------------------
# ✏️ Manual Entry (Now with Pydantic input)
# ------------------------------------------------------------
@app.post("/manual-entry")
def manual_entry(entry: TransactionInput):
    try:
        if not model:
            return JSONResponse({"error": "Model not loaded properly."}, status_code=500)

        df = pd.DataFrame([entry.dict()])

        # Validate required fields
        if not {"date", "amount", "description"}.issubset(df.columns):
            return JSONResponse({"error": "Missing required fields: date, amount, description"}, status_code=400)

        df = prepare_dataframe(df)

        preds = model.predict(df[["description", "amount", "day", "month", "day_of_week"]])
        df["predicted_category"] = preds

        update_excel_database(df)

        entries = df.to_dict(orient="records")
        entries = serialize_dates(entries)

        summary = []
        if os.path.exists(DB_FILE):
            summary_df = pd.read_excel(DB_FILE, sheet_name="Summary")
            summary = summary_df.to_dict(orient="records")

        return JSONResponse({
            "message": "✅ Manual entry processed successfully.",
            "records_added": len(df),
            "entries": entries,
            "summary": summary
        })

    except Exception as e:
        print("❌ Error in /manual-entry:", traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)

# ------------------------------------------------------------
# 🔍 Get All Transactions
# ------------------------------------------------------------
@app.get("/transactions")
def get_transactions(category: str = None):
    if not os.path.exists(DB_FILE):
        return JSONResponse({"error": "Database is empty. Please upload data first."}, status_code=404)

    df = pd.read_excel(DB_FILE, sheet_name="Transactions")
    if category:
        df = df[df["predicted_category"].str.lower() == category.lower()]

    records = df.to_dict(orient="records")
    records = serialize_dates(records)
    return records

# ------------------------------------------------------------
# 🔎 Fetch Transactions Between Dates
# ------------------------------------------------------------
@app.get("/transactions/date-range")
def get_transactions_between_dates(start_date: str, end_date: str, category: str = None):
    """
    Fetch transactions between a start and end date (inclusive).
    Optional: filter by category.
    Example: /transactions/date-range?start_date=2025-11-01&end_date=2025-11-05
    """
    if not os.path.exists(DB_FILE):
        return JSONResponse(
            {"error": "Database is empty. Please upload or add data first."},
            status_code=404
        )

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        if start > end:
            return JSONResponse({"error": "Start date must be before or equal to end date."}, status_code=400)

        df = pd.read_excel(DB_FILE, sheet_name="Transactions")
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

        mask = (df["date"] >= start) & (df["date"] <= end)
        df_filtered = df.loc[mask]

        if category:
            df_filtered = df_filtered[df_filtered["predicted_category"].str.lower() == category.lower()]

        records = df_filtered.to_dict(orient="records")
        records = serialize_dates(records)

        return {
            "message": f"✅ {len(records)} transactions found between {start} and {end}.",
            "count": len(records),
            "data": records
        }

    except ValueError:
        return JSONResponse({"error": "Invalid date format. Use YYYY-MM-DD."}, status_code=400)
    except Exception as e:
        print("❌ Error in /transactions/date-range:", traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)

# ------------------------------------------------------------
# 📊 Get Summary
# ------------------------------------------------------------
@app.get("/summary")
def get_summary():
    if not os.path.exists(DB_FILE):
        return JSONResponse({"error": "Database is empty. Please upload data first."}, status_code=404)

    df = pd.read_excel(DB_FILE, sheet_name="Summary")
    return df.to_dict(orient="records")

# ------------------------------------------------------------
# 📁 Download the Excel Database
# ------------------------------------------------------------
@app.get("/download-db")
def download_db():
    if not os.path.exists(DB_FILE):
        return JSONResponse({"error": "No database found."}, status_code=404)

    return FileResponse(
        DB_FILE,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="FinSmart_DB.xlsx"
    )

# ============================================================
# === REPORT GENERATION: PDF with charts (all/year/month/week) ==
# ============================================================

# Helper: compute date ranges for 'year', 'month', 'week' relative to today
def get_date_range_for_period(period: str):
    today = date.today()
    if period == "all":
        return None, None  # means no filtering
    if period == "year":
        start = date(today.year, 1, 1)
        end = today
        return start, end
    if period == "month":
        start = date(today.year, today.month, 1)
        end = today
        return start, end
    if period == "week":
        start = today - timedelta(days=today.weekday())  # Monday
        end = today
        return start, end
    return None, None

# Helper: load transactions DataFrame safely
def load_transactions_df():
    if not os.path.exists(DB_FILE):
        raise FileNotFoundError("Database not found. Please upload data first.")
    df = pd.read_excel(DB_FILE, sheet_name="Transactions")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date", "amount"])
    return df

# Helper: filter df by date range (inclusive)
def filter_df_by_range(df: pd.DataFrame, start_date, end_date):
    if start_date is None and end_date is None:
        return df
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    return df.loc[mask].copy()

# Chart creators: each returns path to saved PNG
def create_pie_by_category(df: pd.DataFrame, out_path: str, title="Spending by Category"):
    grouped = df.groupby("predicted_category")["amount"].sum().sort_values(ascending=False)
    if grouped.empty:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(grouped.values, labels=grouped.index.tolist(), autopct="%1.1f%%")
    ax.set_title(title)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path

def create_bar_monthly_trend(df: pd.DataFrame, out_path: str, title="Monthly Spending Trend"):
    if df.empty:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    df["ym"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    monthly = df.groupby("ym")["amount"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(monthly["ym"], monthly["amount"])
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Spent")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path

def create_line_weekly_trend(df: pd.DataFrame, out_path: str, title="Daily Spending (last 30 days)"):
    if df.empty:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    df["date_dt"] = pd.to_datetime(df["date"])
    end = df["date_dt"].max()
    start = end - pd.Timedelta(days=30)
    window = df[(df["date_dt"] >= start) & (df["date_dt"] <= end)]
    if window.empty:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.text(0.5, 0.5, "No recent data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    daily = window.groupby(window["date_dt"].dt.date)["amount"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(daily["date_dt"], daily["amount"], marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Spent")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path

def create_top_categories_bar(df: pd.DataFrame, out_path: str, top_n=5, title="Top Categories"):
    if df.empty:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    grouped = df.groupby("predicted_category")["amount"].sum().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.barh(grouped.index[::-1], grouped.values[::-1])
    ax.set_xlabel("Total Spent")
    ax.set_title(title)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path

# Compose PDF using FPDF
def generate_pdf_report(df: pd.DataFrame, period_label: str, out_pdf_path: str):
    tmpdir = tempfile.mkdtemp(prefix="finsmart_report_")
    try:
        total_spent = df["amount"].sum() if not df.empty else 0.0
        count = len(df)

        pie = os.path.join(tmpdir, "pie.png")
        create_pie_by_category(df, pie, title=f"Spending by Category ({period_label})")

        monthly = os.path.join(tmpdir, "monthly.png")
        create_bar_monthly_trend(df, monthly, title=f"Monthly Trend ({period_label})")

        weekly = os.path.join(tmpdir, "weekly.png")
        create_line_weekly_trend(df, weekly, title=f"Daily Trend ({period_label})")

        topcat = os.path.join(tmpdir, "topcat.png")
        create_top_categories_bar(df, topcat, title=f"Top Categories ({period_label})")

        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=15)

        # Title page
        pdf.add_page()
        pdf.set_font("Arial", size=16, style="B")
        pdf.cell(0, 10, "FinSmart Spending Report", ln=True, align="C")
        pdf.ln(4)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 8, f"Period: {period_label}", ln=True)
        pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(6)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 8, f"Total transactions: {count}", ln=True)
        pdf.cell(0, 8, f"Total spent: {total_spent:.2f}", ln=True)
        pdf.ln(6)

        # Insert charts (one per page)
        for img_path, caption in [(pie, "Spending by Category"), (monthly, "Monthly Trend"), (weekly, "Daily Trend"), (topcat, "Top Categories")]:
            pdf.add_page()
            pdf.set_font("Arial", size=12, style="B")
            pdf.cell(0, 8, caption, ln=True)
            pdf.ln(4)
            page_width = 210 - 20  # A4 width minus margins
            pdf.image(img_path, x=10, y=pdf.get_y(), w=page_width)
            pdf.ln(4)

        # Detailed table page (first 100 rows)
        pdf.add_page()
        pdf.set_font("Arial", size=12, style="B")
        pdf.cell(0, 8, "Sample of Transactions (first 100 rows)", ln=True)
        pdf.ln(2)
        pdf.set_font("Arial", size=9)
        headers = ["date", "amount", "description", "predicted_category"]
        col_widths = [30, 30, 80, 40]
        for h, w in zip(headers, col_widths):
            pdf.cell(w, 6, h, border=1)
        pdf.ln()
        max_rows = 100
        for _, row in df.head(max_rows).iterrows():
            pdf.cell(col_widths[0], 6, str(row.get("date", "")), border=1)
            pdf.cell(col_widths[1], 6, f"{row.get('amount', 0):.2f}", border=1)
            desc = str(row.get("description", ""))[:60]
            pdf.cell(col_widths[2], 6, desc, border=1)
            pdf.cell(col_widths[3], 6, str(row.get("predicted_category", "")), border=1)
            pdf.ln()

        pdf.output(out_pdf_path)
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

# Public endpoint: download-report (period param: all/year/month/week)
@app.get("/download-report")
def download_report(period: str = "all"):
    """
    Download a PDF report for spending.
    Query param: period = all | year | month | week
    Example: /download-report?period=month
    """
    try:
        period = period.lower()
        if period not in {"all", "year", "month", "week"}:
            return JSONResponse({"error": "Invalid period. Choose one of: all, year, month, week."}, status_code=400)

        df = load_transactions_df()

        start_date, end_date = get_date_range_for_period(period)
        df_filtered = filter_df_by_range(df, start_date, end_date)

        if period == "all":
            label = "All Time"
        elif period == "year":
            label = f"Year-to-date ({date.today().year})"
        elif period == "month":
            label = f"{date.today().strftime('%B %Y')}"
        else:
            start, end = get_date_range_for_period("week")
            label = f"Week: {start.isoformat()} to {end.isoformat()}"

        unique_name = f"finsmart_report_{period}_{uuid.uuid4().hex[:8]}.pdf"
        tmp_pdf_path = os.path.join(tempfile.gettempdir(), unique_name)
        generate_pdf_report(df_filtered, label, tmp_pdf_path)

        return FileResponse(tmp_pdf_path, media_type="application/pdf", filename=unique_name)

    except FileNotFoundError as fe:
        return JSONResponse({"error": str(fe)}, status_code=404)
    except Exception as e:
        print("❌ Error in /download-report:", traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)
        
# ------------------------------------------------------------
# 📑 Custom Date Range Report Endpoint
# ------------------------------------------------------------
@app.get("/download-report/custom")
def download_custom_report(from_date: str, to_date: str, category: str = None):
    """
    Generate a PDF report for a custom date range.
    Example:
    /download-report/custom?from_date=2025-10-01&to_date=2025-10-31
    /download-report/custom?from_date=2025-10-01&to_date=2025-10-31&category=Food
    """
    try:
        # Parse date strings
        start = datetime.strptime(from_date, "%Y-%m-%d").date()
        end = datetime.strptime(to_date, "%Y-%m-%d").date()

        if start > end:
            return JSONResponse({"error": "from_date must be before or equal to to_date"}, status_code=400)

        # Load transactions
        df = load_transactions_df()

        # Filter by date range
        df_filtered = filter_df_by_range(df, start, end)

        # Optional category filter
        if category:
            df_filtered = df_filtered[df_filtered["predicted_category"].str.lower() == category.lower()]

        # Label for PDF
        label = f"Custom Range: {start.isoformat()} to {end.isoformat()}"
        if category:
            label += f" | Category: {category.title()}"

        # Generate report
        unique_name = f"finsmart_custom_report_{uuid.uuid4().hex[:8]}.pdf"
        tmp_pdf_path = os.path.join(tempfile.gettempdir(), unique_name)
        generate_pdf_report(df_filtered, label, tmp_pdf_path)

        return FileResponse(tmp_pdf_path, media_type="application/pdf", filename=unique_name)

    except ValueError:
        return JSONResponse({"error": "Invalid date format. Use YYYY-MM-DD."}, status_code=400)
    except FileNotFoundError as fe:
        return JSONResponse({"error": str(fe)}, status_code=404)
    except Exception as e:
        print("❌ Error in /download-report/custom:", traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)
