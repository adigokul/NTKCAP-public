# Fix encoding for setup.ps1
$scriptPath = "C:\Users\Nuvo-8003(EMC)\NTKCAP\install\scripts\setup.ps1"
$content = Get-Content $scriptPath -Raw -Encoding UTF8
$Utf8BomEncoding = New-Object System.Text.UTF8Encoding $true
[System.IO.File]::WriteAllText($scriptPath, $content, $Utf8BomEncoding)
Write-Host "Successfully converted setup.ps1 to UTF-8 with BOM" -ForegroundColor Green
