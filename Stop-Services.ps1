# 一键关闭（按端口精准结束）
param(
  [int]$FrontPort = 8501,
  [int]$BackPort = 8000
)

function Stop-ByPort {
  param([int]$Port)
  $procIds = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
             Select-Object -ExpandProperty OwningProcess -Unique
  if ($procIds) {
    foreach ($procId in $procIds) {
      try {
        Stop-Process -Id $procId -Force -ErrorAction Stop
        Write-Host "🛑 Stopped PID ${procId} on port $Port"
      } catch {
        Write-Warning "Failed stopping PID ${procId}: $($_.Exception.Message)"
      }
    }
  } else {
    Write-Host "ℹ️ No process found listening on port $Port."
  }
}

Set-Location $PSScriptRoot
Stop-ByPort $FrontPort
Stop-ByPort $BackPort
Write-Host "✅ Done."
