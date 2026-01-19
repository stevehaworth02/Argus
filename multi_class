# Hierarchical Pipeline - Run Script
# Combines Binary Artifact Filter + Argus Seizure Detector

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "HIERARCHICAL PIPELINE" -ForegroundColor Cyan
Write-Host "Binary Artifact Filter + Argus" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location C:\Users\0218s\Desktop\Ceribell-SZ-DTCTR\multi_class

python hierarchical_pipeline.py `
    --artifact_model ..\full_dataset\models\best_artifact_detector.pth `
    --seizure_model ..\checkpoints\medium_model\best_model.pth `
    --test_data ..\preprocessed\dev.npz `
    --save_dir .\results

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Done! Check .\results\ folder." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
