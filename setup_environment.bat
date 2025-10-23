@echo off
echo 🔧 Configurando entorno virtual para MEG-QC...

REM Navegar al directorio del script
cd /d "%~dp0"

REM Verificar si Python está disponible
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python no encontrado. Por favor instala Python 3.8+
    pause
    exit /b 1
)

REM Crear entorno virtual si no existe
if not exist ".venv" (
    echo 📦 Creando entorno virtual...
    python -m venv .venv
)

REM Activar entorno virtual
echo 🚀 Activando entorno virtual...
call .venv\Scripts\activate

REM Actualizar pip
echo 📦 Actualizando pip...
python -m pip install --upgrade pip

REM Instalar dependencias
echo 📦 Instalando dependencias...
pip install ancpbids
pip install mne
pip install numpy pandas matplotlib scikit-learn seaborn
pip install jupyter notebook

REM Instalar el paquete en modo desarrollo (si es necesario)
if exist "setup.py" (
    echo 📦 Instalando MEG-QC en modo desarrollo...
    pip install -e .
)

REM Verificar instalación
echo ✅ Verificando instalación...
python -c "import ancpbids, mne; print('✅ Todas las dependencias instaladas correctamente')"

echo.
echo 🎉 Entorno configurado exitosamente!
echo 📍 Entorno virtual activado: .venv
echo 📍 Directorio: %CD%
echo.
pause