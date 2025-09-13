# 一键启动（后台运行，不弹窗口）
param(
  [string]$Frontend = "app_front_test.py",
  [int]$FrontPort = 8501,
  [string]$BackendModule = "backend:app",
  [int]$BackPort = 8000
)

function Kill-ByPort {
  param([int]$Port)
  $procIds = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
             Select-Object -ExpandProperty OwningProcess -Unique
  if ($procIds) {
    Write-Host "Port $Port in use by PID(s): $($procIds -join ', '). Stopping..."
    foreach ($procId in $procIds) {
      Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 1
  }
}

# 进入脚本所在目录
Set-Location $PSScriptRoot

# 预清理端口
Kill-ByPort $FrontPort
Kill-ByPort $BackPort

# 日志目录
$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# 启动前端（后台运行，输出到日志）
Start-Process -WindowStyle Hidden -FilePath "powershell" -ArgumentList "-Command","Set-Location `"$PSScriptRoot`"; streamlit run `"$Frontend`" --server.address 0.0.0.0 --server.port $FrontPort *> `"$logDir/frontend.log`""

# 启动后端（后台运行，输出到日志）
Start-Process -WindowStyle Hidden -FilePath "powershell" -ArgumentList "-Command","Set-Location `"$PSScriptRoot`"; uvicorn $BackendModule --host 0.0.0.0 --port $BackPort *> `"$logDir/backend.log`""

Write-Host "✅ Frontend on :$FrontPort and backend on :$BackPort launched (logs in /logs)."
