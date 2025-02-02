python -m venv .venv
call .venv/Scripts/activate.bat
python -m pip install --upgrade pip
pip install -r .setup/requirements.txt

pip install -e packages/analyze
pip install -e packages/fetch
pip install -e packages/utility
pip install -e packages/backtest
pip install -e packages/training