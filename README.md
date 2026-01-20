# Causal Relationship Miner

![Demo](video_2025-12-01_11-52-49.gif)

Extract cause–effect relationships from PDF documents with a simple web app.

## Get started (Windows)

### Option 1 — Easiest (no setup)

1) Download the project
- EITHER click the green “Code” button on GitHub, choose “Download ZIP”, then extract it
- OR use Git (in PowerShell):

```powershell
git clone https://github.com/sciencepresentation/causal_relation_miner.git
cd causal_relation_miner
```

2) Start the app
- Double‑click `launch.bat` inside the project folder

What happens automatically:
- Creates a private Python environment (if missing)
- Installs the required packages (first run may take a few minutes)
- Launches the app in your browser

If you see a message saying Python is not found, please install Python 3.9+ from python.org and run `launch.bat` again.

### Option 2 — Manual (using commands)

1) Download and open the project

```powershell
git clone https://github.com/rasoulnorouzi/causal_relation_miner.git
cd causal_relation_miner
```

2) Create your own environment (one time)

```powershell
python -m venv myenv
```

3) Turn it on and install (one time)

```powershell
.\myenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

4) Start the app

```powershell
streamlit run app.py
```

The app should open in your browser (or copy the URL from the terminal, usually http://localhost:8501).

## Notes

- First time you click “Load Model” in the app, it will download the AI model from the internet. This can take a few minutes.
- If PowerShell blocks the activation script, run this once in the same terminal, then try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## What’s inside

- `app.py` – the Streamlit web app
- `utils/` – PDF processing and search helpers
- `requirements.txt` – list of required packages
