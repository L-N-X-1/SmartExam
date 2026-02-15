@echo off
echo ============================================
echo    SMART EXAM - Demarrage
echo ============================================
echo.

cd /d "%~dp0"

echo Activation environnement virtuel...
call venv\Scripts\activate.bat

echo.
echo Verification Python...
python --version
echo.

echo Verification modules...
python -c "from docx import Document; print('  python-docx : OK')" 2>nul || echo   python-docx : MANQUANT
python -c "import streamlit; print('  streamlit : OK')" 2>nul || echo   streamlit : MANQUANT
echo.

echo Lancement de l application...
echo URL: http://localhost:8501
echo.
echo Appuyez sur Ctrl+C pour arreter
echo.

python -m streamlit run app.py

pause