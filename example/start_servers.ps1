# Start servers in conda environment 507
# Usage: .\start_servers.ps1

Write-Host "Activating conda environment 507..." -ForegroundColor Green
conda activate 507

Write-Host "`nStarting backend server on port 3001..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "conda activate 507; cd backend; npm start" -WindowStyle Normal

Start-Sleep -Seconds 2

Write-Host "Starting frontend server on port 5173..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "conda activate 507; cd frontend; npm run dev" -WindowStyle Normal

Write-Host "`n✅ Servers starting in separate windows!" -ForegroundColor Green
Write-Host "Backend: http://localhost:3001" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:5173" -ForegroundColor Cyan
Write-Host "`nOpen http://localhost:5173 in your browser to view the app." -ForegroundColor White

