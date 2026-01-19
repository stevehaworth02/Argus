@echo off
echo ========================================
echo HIERARCHICAL PIPELINE
echo Binary Artifact Filter + Argus
echo ========================================
echo.

cd C:\Users\0218s\Desktop\Ceribell-SZ-DTCTR\multi_class

python hierarchical_pipeline.py ^
    --artifact_model ..\full_dataset\models\best_artifact_detector.pth ^
    --seizure_model ..\checkpoints\medium_model\best_model.pth ^
    --test_data ..\preprocessed\dev.npz ^
    --save_dir .\results

echo.
echo ========================================
echo Done! Check results folder.
echo ========================================
pause
