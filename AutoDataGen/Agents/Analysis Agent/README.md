# AutoDataGen — Agent 1 (Dataset Analysis Agent)

Pehla working version — **bina LangGraph ke**. CSV upload karo, deterministic
Pandas/NumPy tools chalte hain, structured JSON report banta hai, aur LLM sirf
usko human-readable explanation mein convert karta hai.

```
Upload -> FastAPI -> type-specific diagnostics -> JSON -> PostgreSQL audit report
```

## Folder structure

```
autodatagen/
├── backend/
│   ├── main.py                 # FastAPI app, /upload endpoint
│   ├── agents/
│   │   └── analysis_agent.py   # orchestrates all 6 tools + quality score + LLM summary
│   ├── tools/
│   │   ├── dataset_info.py     # rows/columns
│   │   ├── column_analysis.py  # dtype classification, unique values
│   │   ├── missing_values.py   # missing %, duplicates
│   │   ├── statistics.py       # mean/median/std/min/max/quartiles
│   │   ├── outliers.py         # IQR-based outlier detection
│   │   └── class_analysis.py   # target suggestion + class imbalance
│   ├── models/
│   │   └── state.py            # pydantic schema of the final report
│   └── utils/
│       └── file_handler.py     # upload validation + safe loading
├── uploads/                    # uploaded datasets land here
├── reports/                    # local sample/report assets
├── requirements.txt
└── README.md
```

## Setup

```bash
cd autodatagen
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env          # Windows PowerShell: Copy-Item .env.example .env
```

The application loads local secrets from `.env`. Never commit `.env`; it is
ignored by Git. Put real API keys and database credentials only in that file
or in your deployment platform's secret settings.

The agent accepts CSV, TXT/MD/LOG, single PNG/JPG/JPEG/WEBP/BMP/TIFF images,
and ZIP image datasets.
CSV files receive the full Pandas/NumPy/scikit-learn-oriented tabular report:
logical column types, missing values, duplicates, statistics, IQR outliers,
target suggestions, class imbalance, and a quality score. Text files receive
length and vocabulary statistics. Images are checked with Pillow and OpenCV
for integrity, dimensions, format, and channel information.
For image datasets, upload a ZIP. Put class labels in folders, for example
`cats/cat-01.jpg` and `dogs/dog-01.jpg`. The report counts valid/corrupt images,
duplicate files, labels, class imbalance, dimensions, formats, and channels.

## PostgreSQL report storage

Create an empty PostgreSQL database and set its connection string before
starting the API. The table is created automatically on the first successful
upload:

```powershell
# Either add DATABASE_URL to .env, or set it for the current PowerShell session:
$env:DATABASE_URL = "postgresql://postgres:password@localhost:5432/autodatagen"
```

Each report is stored as JSONB in `dataset_analysis_reports` with filename,
dataset type, generated ID, and timestamp. Retrieve an audit/retry report with
`GET /reports/{report_id}`. Without `DATABASE_URL`, analysis still runs but the
response reports `persistence.saved: false`.

## (Optional) LLM explanation

Agent tools ke bina LLM ke bhi chalega — tab ek templated fallback summary
banta hai. Agar Claude se explanation chahiye:

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # Windows: set ANTHROPIC_API_KEY=...
```

For local development, prefer adding `ANTHROPIC_API_KEY=sk-ant-...` to `.env`.
The app falls back to the deterministic summary when this key is missing.

## Upload to GitHub

Run these commands from the `autodatagen` folder:

```powershell
git init
git add .
git status                         # confirm .env, venv, and uploads are absent
git commit -m "Initial project setup"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Create an empty repository on GitHub first. If a secret was ever committed,
rotate/revoke it immediately, remove it from Git history, and create a new key;
deleting the file in a later commit does not make the old key safe.

## Run

```bash
cd backend
uvicorn main:app --reload --port 8000
```

## Test it

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@/path/to/your_dataset.csv"

# with an explicit target column:
curl -X POST "http://localhost:8000/upload?target_column=Churn" \
  -F "file=@/path/to/your_dataset.csv"

# retrieve a stored report for a later retry iteration:
curl "http://localhost:8000/reports/1"
```

Response shape (`report` field):

```json
{
  "dataset": { "rows": 7043, "columns": 21, "column_names": [...] },
  "columns": {
    "numerical": [...], "binary": [...], "categorical": [...],
    "datetime": [...], "text": [...], "unique_values": {...}
  },
  "missing_values": { "gender": { "count": 120, "percentage": 1.7 } },
  "duplicates": 43,
  "statistics": { "age": { "mean": 29.4, "median": 28, ... } },
  "outliers": { "salary": 32 },
  "target_suggestions": ["Churn", "Outcome"],
  "target": {
    "column": "Churn", "classes": 2,
    "distribution": { "No": 5174, "Yes": 1869 },
    "imbalanced": true, "majority_share_percent": 73.5
  },
  "quality_score": 78,
  "llm_summary": "..."
}
```

## Manual checklist

1. Install PostgreSQL and create the `autodatagen` database.
2. Set `DATABASE_URL` in the shell or a secrets manager; do not commit it.
3. Run `pip install -r requirements.txt` and start Uvicorn.
4. Upload a small CSV, text file, and valid image, then confirm `report_id` is
  returned and `GET /reports/{report_id}` works.
5. Set `ANTHROPIC_API_KEY` only if an LLM-written summary is required; all
  measurements and the quality score are deterministic without it.
