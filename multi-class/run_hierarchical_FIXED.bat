@echo off
echo ========================================
echo HIERARCHICAL PIPELINE - FIXED VERSION
echo Binary Artifact Filter + Argus
echo ========================================
echo.

cd C:\Users\0218s\Desktop\Ceribell-SZ-DTCTR\multi_class

echo Copying model.py to modules folder...
if not exist "..\modules" mkdir "..\modules"
copy /Y model.py ..\modules\model.py

echo.
echo Running hierarchical pipeline...
echo.

python hierarchical_pipeline_FIXED.py ^
    --artifact_model ..\full_dataset\models\best_artifact_detector.pth ^
    --seizure_model ..\checkpoints\medium_model\best_model.pth ^
    --test_data ..\preprocessed\dev.npz ^
    --save_dir .\results

echo.
echo ========================================
echo Done! Check results folder.
echo ========================================
pause
