## A-Insights – Employee Attrition Analytics System

A-Insights is a final-year project that combines a **FastAPI backend**, **React/TypeScript frontend**, and a **machine learning attrition model** to help HR teams analyse employee turnover risk at snapshot points in time.

The project is designed to be **reproducible, explainable, and academically defensible**: the data pipeline, model training, and inference are all explicit and documented.

---

### 1. System architecture

- **Backend** (`ainsights-backend`)
  - FastAPI application with JWT-based auth
  - SQLite database via SQLAlchemy models (`users`, `departments`, `snapshots`, `employees`, `performance_records`, `data_uploads`)
  - CSV upload pipeline that normalises HR data into per-snapshot employees and departments
  - ML training script (`ml_training.py`) that trains a turnover model on the IBM attrition dataset
  - ML inference service (`ml_service.py`) that:
    - Reads employee + extra feature columns from the DB
    - Aligns them to the training schema
    - Runs probability-based attrition risk scoring per snapshot
    - Computes feature coverage diagnostics

- **Frontend** (`ainsights-front`)
  - React 18 + TypeScript + Vite
  - Tailwind CSS + shadcn/ui for a professional dashboard UI
  - Authenticated layout with sidebar navigation
  - Screens for dashboard KPIs, snapshot management, data upload, employee list/detail, risk summaries, and simple settings/account pages

---

### 2. Key implemented features

- **Authentication & accounts**
  - Register, login, logout, password reset via email-token style flow
  - Per-user isolation of departments, snapshots, employees, and uploads

- **Snapshots & HR data pipeline**
  - Create named monthly/yearly snapshots
  - Upload HR CSV for a specific snapshot via `/api/v1/uploads`
  - Backend cleaning/normalisation:
    - Normalises column names (e.g. `EmployeeNumber`, `JobRole`, `MonthlyIncome`)
    - Validates required identifiers and departments
    - Converts tenure in years → months, salary to numeric, and performance scores
    - Creates departments if needed and employees bound to the chosen snapshot

- **Machine learning model**
  - Training on the IBM “WA_Fn-UseC_-HR-Employee-Attrition.csv” dataset
  - Full sklearn pipeline (preprocess + model) persisted as `turnover_model.pkl`
  - Model expects the same IBM-style input columns in production; these are preserved in `Employee.extra_features`
  - Feature importance summary exported for explanation

- **Prediction & diagnostics**
  - For any snapshot:
    - Predicts per-employee attrition risk probabilities and labels (`Low` / `Medium` / `High`)
    - Aggregated risk summary by department and overall
  - **Feature coverage diagnostics**:
    - Per-employee coverage ratio (what fraction of model features are populated)
    - Snapshot-level diagnostics endpoint describing coverage statistics
    - Safety behaviour: if all employees are very low-coverage, the prediction endpoints fail with a clear error instead of pretending results are meaningful

- **Frontend**
  - Dashboard with snapshot-based analytics and risk summaries
  - Snapshot list, duplication, compare, and analytics views
  - Upload page describing CSV expectations and driving the backend pipeline
  - Employee list and detail views tied to real database records
  - Settings page that honestly documents which configuration is in scope

---

### 3. Local setup and running

#### 3.1 Prerequisites

- Python 3.11+
- Node.js 18+ and npm
- Git

#### 3.2 Backend setup (`ainsights-backend`)

1. Create and activate a virtual environment (recommended):

   ```bash
   cd ainsights-backend
   python -m venv venv
   source venv/Scripts/activate  # Windows PowerShell: venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment (development):

   - Copy `.env.example` to `.env` (if not already present) and adjust values as needed.
   - By default the backend uses a local SQLite database file `ainsights.db` in the backend folder.

4. Run the FastAPI server:

   ```bash
   uvicorn main:app --reload
   ```

   The API will be available at `http://127.0.0.1:8000` with interactive docs at `/docs`.

#### 3.3 Frontend setup (`ainsights-front`)

1. Install dependencies:

   ```bash
   cd ainsights-front
   npm install
   ```

2. Configure environment:

   - Create `.env.local` (or `.env`) with at least:

   ```bash
   VITE_API_URL=http://127.0.0.1:8000
   ```

3. Run the dev server:

   ```bash
   npm run dev
   ```

   The app will be available at the URL printed by Vite (commonly `http://localhost:5173`).

---

### 4. Model training and reproducibility

- Training script: `ainsights-backend/ml_training.py`
- Typical usage from the backend folder:

```bash
python ml_training.py \
  --csv-path ../WA_Fn-UseC_-HR-Employee-Attrition.csv \
  --target-column Attrition
```

This will:

- Train and cross-validate candidate models on the IBM dataset
- Select the best model by ROC-AUC
- Save a single pipeline artifact as `turnover_model.pkl` in `ainsights-backend`
- Write top-10 feature importance to `feature_importance.json`

The prediction endpoints then load this pipeline and enforce alignment between:

- Training feature schema (IBM columns minus IDs/target), and
- Production inputs built from `Employee` attributes + `extra_features`.

Feature coverage diagnostics help you justify model usage in your report and demo.

---

### 5. Environment variables and secrets

- **Backend** (`ainsights-backend/.env.example`)
  - `AINSIGHTS_SECRET_KEY` – JWT signing key (set a strong value in real deployments)
  - `AINSIGHTS_MIN_FEATURE_COVERAGE` – optional float threshold for ML coverage safety (default `0.6`)
  - Optional database or model-path overrides can be added if needed.

- **Frontend** (`ainsights-front/.env.local` or `.env`)
  - `VITE_API_BASE_URL` – base URL of the FastAPI backend.

Notes:

- Do **not** commit real production secrets; use example files and local overrides instead.
- The provided `.env` files in this repository are for local development and can be rotated for any external deployment.

---

### 6. Known limitations and future work

- **Data scope**
  - The ML model is trained on the IBM attrition dataset; real-world HR datasets may require additional feature engineering and re-training.

- **Frontend coverage**
  - The core flows (auth, snapshots, upload, employees, risk summaries) are implemented.
  - Some advanced settings and narrative “AI insight” features are described conceptually but not fully implemented in this submission.

- **Scalability & deployment**
  - The project targets a single-node academic demo with SQLite; in production you would typically migrate to Postgres and add proper worker queues for heavy ML jobs.

These limitations are intentional trade-offs for a clear, inspectable final-year project implementation.

---

### 7. How to read the code during inspection

- **Start with the backend**:
  - `main.py` – API entrypoint, auth, CRUD routes
  - `models.py` / `schemas.py` – database and Pydantic models
  - `upload_routes.py` – CSV ingestion and cleaning
  - `snapshot_routes.py` – snapshot analytics and prediction endpoints
  - `ml_training.py` / `ml_service.py` – ML training and inference

- **Then review the frontend**:
  - `src/components/layout/MainLayout.tsx` + `Sidebar.tsx` – overall shell
  - `src/pages/Dashboard.tsx`, `Snapshots.tsx`, `Upload.tsx`, `EmployeeList.tsx`, `EmployeeDetail.tsx`

Together these files show how raw HR data flows from CSV → database → model → visual analytics and risk summaries.

---

### 8. Submission checklist (for packaging this project)

Before creating a submission zip or pushing to a public repository:

1. **Do include**
   - `ainsights-backend` source code and `requirements.txt`
   - `ainsights-front` source code and `package.json` / `package-lock.json`
   - `turnover_model.pkl` and `feature_importance.json` (or clear instructions for re-training)
   - The IBM attrition CSV used for training (if permitted by your institution) or a link to its public source
   - This root `README.md` and the frontend `README.md`

2. **Exclude or regenerate**
   - Local virtual environments (`venv/`, `.venv/`) and `node_modules/` folders
   - Any build artefacts (`dist/`, `dist-ssr/`)
   - OS/editor files (`.DS_Store`, `.idea/`, `.vscode/` except optional shared config)
   - Large log files, if any

3. **Environment files**
   - Keep `.env.example` in `ainsights-backend` and document variables.
   - Ensure real `.env` files and any production secrets are **not** included in the submitted zip.
   - For examiners, mention in your report that `.env.example` shows all required variables.

4. **Quick run test**
   - From a clean clone:
     - Install backend and frontend dependencies.
     - Run the training script (or ensure `turnover_model.pkl` is present).
     - Start backend (`uvicorn`) and frontend (`npm run dev`).
   - Verify that you can register, create a snapshot, upload a CSV, and obtain predictions and analytics.

---

### 9. Demo startup checklist (first run)

When preparing for a live demo or inspection:

1. **Backend**
   - Ensure you are using a clean virtual environment and have run:
     ```bash
     cd ainsights-backend
     pip install -r requirements.txt
     ```
   - If `turnover_model.pkl` is missing or outdated, train the model:
     ```bash
     python ml_training.py --csv-path ../WA_Fn-UseC_-HR-Employee-Attrition.csv --target-column Attrition
     ```
   - Start the API server:
     ```bash
     uvicorn main:app --reload
     ```
   - If you hit prediction errors about a missing model or feature schema, check:
     - That `turnover_model.pkl` exists in `ainsights-backend`.
     - That the training CSV path and target column were correct.

2. **Frontend**
   - Confirm the frontend is pointing at the correct backend URL:
     - In `ainsights-front/.env.local` (or `.env`), set:
       ```bash
       VITE_API_URL=http://127.0.0.1:8000
       ```
   - Install dependencies and run the dev server:
     ```bash
     cd ainsights-front
     npm install
     npm run dev
     ```

3. **First-run flow**
   - Register a user account via the UI (or `/auth/register`).
   - Create a snapshot for the current period.
   - Upload the IBM-style attrition CSV for that snapshot.
   - Navigate to the dashboard and snapshot views to confirm:
     - Employees are visible for the snapshot.
     - Predictions and risk summaries are available (if the model is trained).
     - Feature coverage and model summary endpoints respond without server errors.


