# Compila a tese numa copia de build (o symlink "figuras/" da escrita vira arquivo-texto no Windows).
# Uso, a partir de qualquer pasta:  powershell -File reprodutibilidade\build_tese.ps1
# Saida: reprodutibilidade\build_tese\main.pdf e um resumo (paginas, erros, overfull, referencias indefinidas).
param([string]$Repo = (Split-Path $PSScriptRoot -Parent))
$ErrorActionPreference = "Continue"

# MiKTeX: usa o PATH; se nao achar, tenta a instalacao por usuario.
$bin = Split-Path (Get-Command pdflatex.exe -ErrorAction SilentlyContinue).Source -ErrorAction SilentlyContinue
if (-not $bin) { $bin = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64" }
if (-not (Test-Path "$bin\pdflatex.exe")) { throw "pdflatex nao encontrado (instale o MiKTeX ou ponha no PATH)" }

$src = Join-Path $Repo "Escrita Final\texto"
$build = Join-Path $PSScriptRoot "build_tese"
if (Test-Path $build) { Remove-Item $build -Recurse -Force }
New-Item -ItemType Directory $build | Out-Null
Get-ChildItem $src -File | Where-Object { $_.Name -ne "figuras" -and $_.Extension -in ".tex", ".bib", ".bst" } | Copy-Item -Destination $build
Copy-Item (Join-Path $Repo "Escrita Final\figuras") (Join-Path $build "figuras") -Recurse
Push-Location $build
try {
    cmd /c "`"$bin\pdflatex.exe`" -interaction=nonstopmode --enable-installer main.tex > pass1.log 2>&1"
    cmd /c "`"$bin\bibtex.exe`" main > bibtex.log 2>&1"
    cmd /c "`"$bin\pdflatex.exe`" -interaction=nonstopmode main.tex > pass2.log 2>&1"
    cmd /c "`"$bin\pdflatex.exe`" -interaction=nonstopmode main.tex > pass3.log 2>&1"
} finally { Pop-Location }
$log = Get-Content (Join-Path $build "main.log") -Raw
$erros = ([regex]::Matches($log, "(?m)^! ")).Count
$overfull = ([regex]::Matches($log, "Overfull \\hbox")).Count
$undef = ([regex]::Matches($log, "(Citation|Reference) .* undefined")).Count
$pdf = Join-Path $build "main.pdf"
$paginas = if (Test-Path $pdf) { (python -c "import fitz,sys;print(fitz.open(sys.argv[1]).page_count)" $pdf) } else { 0 }
"paginas=$paginas erros=$erros overfull=$overfull indefinidas=$undef"
"Para atualizar o PDF versionado: Copy-Item '$pdf' '$src\main.pdf' -Force"
Get-Content (Join-Path $build "bibtex.log") | Select-String "Warning|error" | Select-Object -First 10
