@echo off
chcp 65001 >nul
echo ============================================================
echo PaddlePE End-to-End Client Mode Verification
echo ============================================================
echo.
echo [1/5] Starting inference server in background...
start /B python -m paddlepe.server --model fcpe --port 28789
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Server failed to start
    exit /b 1
)

echo [2/5] Waiting for server to be ready...
:WAIT
timeout /t 3 /nobreak >nul
python -c "import urllib.request, json; resp = urllib.request.urlopen('http://127.0.0.1:28789/health', timeout=5); d=json.loads(resp.read()); print('  Health:', d); assert d.get('status')=='ok'; exit(0)" 2>nul
if %ERRORLEVEL% NEQ 0 (
    set /a tries+=1
    if !tries! LSS 20 goto WAIT
    echo [FAIL] Server did not start within 60 seconds
    exit /b 1
)
echo   [OK] Server is ready

echo [3/5] Testing RemotePE via URL...
python -c ^
    "from paddlepe.remote import RemotePE; import numpy as np; " ^
    "pe = RemotePE(model='fcpe', url='http://127.0.0.1:28789', auto_shutdown=False); " ^
    "wav = np.sin(2*np.pi*440*np.linspace(0,0.3,4800)).astype(np.float32); " ^
    "f0, conf = pe.infer(wav, 16000); " ^
    "print('  [OK] infer -> f0:', f0.shape); " ^
    "print('  [OK] First few values:', f0[:5])"
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Inference failed
    exit /b 1
)

echo [4/5] Testing ClientPE.list_models via temp server...
python -c ^
    "from paddlepe.client import ClientPE; " ^
    "models = ClientPE.list_models(); " ^
    "print('  [OK] models:', models); " ^
    "assert 'fcpe' in models"
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] list_models failed
    exit /b 1
)

echo [5/5] Shutting down server...
python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:28789/shutdown', data=b'{}', timeout=3)" 2>nul
echo   [OK] Shutdown signal sent

echo.
echo ============================================================
echo [PASS] All Client mode tests passed!
echo ============================================================
