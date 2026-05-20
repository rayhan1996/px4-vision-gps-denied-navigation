@echo off

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing project requirements...
pip install -r requirements.txt

echo.
echo ====================================
echo Environment setup complete.
echo To activate later use:
echo venv\Scripts\activate
echo ====================================

pause
