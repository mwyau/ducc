param(
  [Parameter(Mandatory=$true)]
  [ValidateSet('avx', 'avx512')]
  [string] $Isa
)

$ErrorActionPreference = 'Stop'

function Invoke-Checked {
  param(
    [Parameter(Mandatory=$true)] [string] $FilePath,
    [Parameter(Mandatory=$false)] [string[]] $ArgumentList = @()
  )
  & $FilePath @ArgumentList
  if ($LASTEXITCODE -ne 0) {
    throw "command failed ($LASTEXITCODE): $FilePath $($ArgumentList -join ' ')"
  }
}

function Append-Summary {
  param([Parameter(Mandatory=$true)] [string] $Text)
  Add-Content -Path $env:GITHUB_STEP_SUMMARY -Value $Text
}

function Print-CMakeConfig {
  param(
    [Parameter(Mandatory=$true)] [string] $BuildDir,
    [Parameter(Mandatory=$true)] [string] $Label
  )
  $cache = Get-ChildItem -Path $BuildDir -Filter CMakeCache.txt -Recurse |
    Select-Object -First 1
  if ($null -eq $cache) {
    throw "no CMakeCache.txt found below $BuildDir"
  }
  Write-Host "[$Label] CMake cache: $($cache.FullName)"
  Get-Content $cache.FullName |
    Select-String '^(CMAKE_CXX_COMPILER:|DUCC0_ARCH_FLAGS:)' |
    ForEach-Object { Write-Host "[$Label] $($_.Line)" }
  $compilerConfig = Get-ChildItem -Path $BuildDir -Filter CMakeCXXCompiler.cmake -Recurse |
    Select-Object -First 1
  if ($null -eq $compilerConfig) {
    throw "no CMakeCXXCompiler.cmake found below $BuildDir"
  }
  Get-Content $compilerConfig.FullName |
    Select-String 'set\(CMAKE_CXX_COMPILER(_ID|_VERSION)? ' |
    ForEach-Object { Write-Host "[$Label] $($_.Line)" }
}

function Install-Ducc {
  param(
    [Parameter(Mandatory=$true)] [string] $PythonExe,
    [Parameter(Mandatory=$true)] [string] $SourceDir,
    [Parameter(Mandatory=$true)] [string] $BuildDir,
    [Parameter(Mandatory=$true)] [string] $Label,
    [Parameter(Mandatory=$false)] [string] $DuccCFlags = ''
  )
  $hadDuccCFlags = Test-Path Env:DUCC0_CFLAGS
  $savedDuccCFlags = $env:DUCC0_CFLAGS
  try {
    if ([string]::IsNullOrWhiteSpace($DuccCFlags)) {
      Remove-Item Env:DUCC0_CFLAGS -ErrorAction SilentlyContinue
    }
    else {
      $env:DUCC0_CFLAGS = $DuccCFlags
    }
    Write-Host "Installing $Label from $SourceDir with $($env:CMAKE_ARGS) and DUCC0_CFLAGS=$DuccCFlags"
    Push-Location $SourceDir
    try {
      Invoke-Checked $PythonExe @('-m', 'pip', 'install', '-v', '--no-build-isolation',
        '--no-deps', '--no-cache-dir', "--config-settings=build-dir=$BuildDir", $SourceDir)
    }
    finally {
      Pop-Location
    }
  }
  finally {
    if ($hadDuccCFlags) { $env:DUCC0_CFLAGS = $savedDuccCFlags }
    else { Remove-Item Env:DUCC0_CFLAGS -ErrorAction SilentlyContinue }
  }
  Print-CMakeConfig -BuildDir $BuildDir -Label $Label
  Invoke-Checked $PythonExe @('-c', 'import ducc0; print("ducc0=", ducc0.__file__)')
}

function Compile-SimdProbe {
  param(
    [Parameter(Mandatory=$true)] [string] $SourceDir,
    [Parameter(Mandatory=$true)] [string] $ExpectedMacro,
    [Parameter(Mandatory=$true)] [string] $Label,
    [Parameter(Mandatory=$true)] [string] $ArchFlag,
    [Parameter(Mandatory=$true)] [string] $Clang,
    [Parameter(Mandatory=$true)] [string] $Root,
    [Parameter(Mandatory=$true)] [string] $OutDir
  )
  $exe = Join-Path $OutDir "$Label-simd-probe.exe"
  $args = @('/nologo', '/O2', '/EHsc', '/std:c++17', $ArchFlag,
    "/D$ExpectedMacro", "/I$SourceDir\src",
    (Join-Path $Root 'ci\ducc_simd_probe.cpp'), "/Fe:$exe")
  Invoke-Checked $Clang $args
  $output = (& $exe | Out-String).Trim()
  if ($LASTEXITCODE -ne 0) { throw "$Label SIMD probe failed" }
  Write-Host "[$Label] $output"
  if ($output -notmatch 'DUCC0_NO_SIMD=0') {
    throw "$Label SIMD probe did not prove DUCC0 SIMD is active"
  }
  return $output
}

function Compile-HsumBenchmark {
  param(
    [Parameter(Mandatory=$true)] [string] $SourceDir,
    [Parameter(Mandatory=$true)] [string] $Label,
    [Parameter(Mandatory=$true)] [string] $Clang,
    [Parameter(Mandatory=$true)] [string] $Root,
    [Parameter(Mandatory=$true)] [string] $OutDir,
    [Parameter(Mandatory=$true)]
    [ValidateSet('sse3', 'sse2')]
    [string] $Mode
  )
  $exe = Join-Path $OutDir "$Label-$Mode-hsum-sse.exe"
  $asm = Join-Path $OutDir "$Label-$Mode-hsum-sse.asm"
  if ($Mode -eq 'sse3') {
    $modeArgs = @('/clang:-msse3', '/DDUCC_HSUM_EXPECT_SSE3')
  }
  else {
    $modeArgs = @('/clang:-msse2', '/clang:-mno-sse3', '/DDUCC_HSUM_EXPECT_SSE2')
  }
  $args = @('/nologo', '/O2', '/EHsc', '/std:c++17', '/FAcs', "/Fa:$asm",
    '/clang:-mno-avx') + $modeArgs + @("/I$SourceDir\src",
    (Join-Path $Root 'ci\hsum_sse_benchmark.cpp'), "/Fe:$exe")
  Invoke-Checked $Clang $args
  $assembly = Get-Content -Raw $asm
  $hasHadd = $assembly -match '(?i)haddps|vhaddps'
  if ($Mode -eq 'sse3' -and -not $hasHadd) {
    throw "$Label SSE3 hsum assembly did not retain haddps"
  }
  if ($Mode -eq 'sse2' -and $hasHadd) {
    throw "$Label SSE2-only hsum assembly contains a horizontal-add instruction"
  }
  Write-Host "[$Label/$Mode] hsum assembly has_hadd=$hasHadd"
  return $exe
}

function Compile-ShtProbe {
  param(
    [Parameter(Mandatory=$true)] [string] $SourceDir,
    [Parameter(Mandatory=$true)] [string] $Label,
    [Parameter(Mandatory=$true)] [string] $Clang,
    [Parameter(Mandatory=$true)] [string] $Root,
    [Parameter(Mandatory=$true)] [string] $OutDir,
    [Parameter(Mandatory=$true)] [string] $ArchFlag,
    [Parameter(Mandatory=$true)] [int] $Variant
  )
  $exe = Join-Path $OutDir "$Label-sht-simd-probe-$Variant.exe"
  $asm = Join-Path $OutDir "$Label-sht-simd-probe-$Variant.asm"
  if ($Variant -eq 0) {
    $variantArgs = @()
  }
  else {
    $variantArgs = @("/DDUCC0_SHT_AVX512_REDUCTION_VARIANT=$Variant")
  }
  $args = @('/nologo', '/O2', '/EHsc', '/std:c++17', '/FAcs', "/Fa:$asm",
    $ArchFlag) + $variantArgs + @("/I$SourceDir\src",
    (Join-Path $Root 'ci\sht_simd_probe.cpp'), "/Fe:$exe")
  Invoke-Checked $Clang $args
  $assembly = Get-Content -Raw $asm
  $hasZmm = $assembly -match '(?i)zmm'
  if ($Variant -eq 1 -and -not $hasZmm) {
    throw 'candidate A SHT AVX-512 probe did not emit ZMM code'
  }
  if ($Variant -eq 2 -and
      ($assembly -notmatch '(?i)zmm' -or $assembly -notmatch '(?i)vextractf64x4')) {
    throw 'candidate B SHT AVX-512 probe did not emit the expected fold sequence'
  }
  Write-Host "[$Label/variant=$Variant] SHT probe assembly has_zmm=$hasZmm"
  Invoke-Checked $exe
  return $exe
}

function Run-PythonCase {
  param(
    [Parameter(Mandatory=$true)] [string] $PythonExe,
    [Parameter(Mandatory=$true)] [string] $Root,
    [Parameter(Mandatory=$true)] [string] $Case,
    [Parameter(Mandatory=$true)] [string] $OutputDir,
    [Parameter(Mandatory=$true)] [int] $Warmups,
    [Parameter(Mandatory=$true)] [int] $Repetitions
  )
  New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
  Invoke-Checked $PythonExe @((Join-Path $Root 'ci\simd_validation.py'), 'run',
    '--case', $Case, '--output', $OutputDir, '--warmups', $Warmups,
    '--repetitions', $Repetitions, '--threads', '1')
}

function Compare-PythonCase {
  param(
    [Parameter(Mandatory=$true)] [string] $BaselinePython,
    [Parameter(Mandatory=$true)] [string] $CandidatePython,
    [Parameter(Mandatory=$true)] [string] $Root,
    [Parameter(Mandatory=$true)] [string] $Case,
    [Parameter(Mandatory=$true)] [string] $WorkRoot,
    [Parameter(Mandatory=$true)] [string] $Label,
    [Parameter(Mandatory=$true)] [int] $Repetitions
  )
  $baseRuns = @()
  $candidateRuns = @()
  for ($round=0; $round -lt 4; ++$round) {
    if (($round % 2) -eq 0) { $order = @('candidate', 'baseline') }
    else { $order = @('baseline', 'candidate') }
    foreach ($which in $order) {
      $runDir = Join-Path $WorkRoot "api-$Case-$Label-$which-$round"
      if ($which -eq 'baseline') {
        Run-PythonCase -PythonExe $BaselinePython -Root $Root -Case $Case -OutputDir $runDir -Warmups 2 -Repetitions $Repetitions
        $baseRuns += $runDir
      }
      else {
        Run-PythonCase -PythonExe $CandidatePython -Root $Root -Case $Case -OutputDir $runDir -Warmups 2 -Repetitions $Repetitions
        $candidateRuns += $runDir
      }
    }
  }
  $summary = Join-Path $WorkRoot "$Case-$Label-summary.md"
  $compareArgs = @((Join-Path $Root 'ci\simd_validation.py'), 'compare',
    '--case', $Case, '--summary', $summary)
  foreach ($runDir in $baseRuns) { $compareArgs += @('--baseline', $runDir) }
  foreach ($runDir in $candidateRuns) { $compareArgs += @('--candidate', $runDir) }
  & $CandidatePython @compareArgs
  $compareExit = $LASTEXITCODE
  $summaryText = ''
  if (Test-Path $summary) {
    $summaryText = Get-Content -Raw $summary
    $summaryText | Add-Content $env:GITHUB_STEP_SUMMARY
  }
  if ($compareExit -ne 0) { throw "focused $Case comparison for $Label failed ($compareExit)" }
  $script:LastComparisonVerified = [bool]($summaryText -match '(?i)\|\s*verified improvement\s*\|')
}

$root = Split-Path -Parent $PSScriptRoot
$baselineSha = $env:BASELINE_SHA
if ([string]::IsNullOrWhiteSpace($baselineSha)) {
  throw 'BASELINE_SHA is not set'
}

$candidateSha = (& git -C $root rev-parse HEAD | Out-String).Trim()
if ($LASTEXITCODE -ne 0) { throw 'could not resolve candidate HEAD' }
$resolvedBaseline = (& git -C $root rev-parse --verify "$baselineSha^{commit}" | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $resolvedBaseline -ne $baselineSha) {
  throw "required baseline is not locally resolvable: $baselineSha"
}

$workRoot = Join-Path $env:RUNNER_TEMP "ducc-simd-$Isa"
New-Item -ItemType Directory -Force -Path $workRoot | Out-Null
$baselineSource = Join-Path $workRoot 'baseline-source'
Invoke-Checked 'git' @('-C', $root, 'worktree', 'add', '--detach', $baselineSource, $baselineSha)

$python = (Get-Command python.exe -ErrorAction Stop).Source
$clang = (Get-Command clang-cl.exe -ErrorAction Stop).Source
$cmake = (Get-Command cmake.exe -ErrorAction Stop).Source

Write-Host "candidate_sha=$candidateSha"
Write-Host "baseline_sha=$baselineSha"
Write-Host "compiler_path=$clang"
Write-Host "compiler_version="
& $clang '--version'
if ($LASTEXITCODE -ne 0) { throw 'could not query clang-cl' }
Write-Host "cmake_version="
Invoke-Checked $cmake @('--version')
Write-Host "python_version="
Invoke-Checked $python @('--version')
Write-Host "architecture_case=$Isa"

$isaProbeSource = Join-Path $root 'ci\x86_isa_probe.cpp'
$isaProbe = Join-Path $workRoot 'x86-isa-probe.exe'
Invoke-Checked $clang @('/nologo', '/O2', '/EHsc', '/std:c++17', '/clang:-mxsave', $isaProbeSource,
  "/Fe:$isaProbe")
$isaOutput = (& $isaProbe | Out-String).Trim()
if ($LASTEXITCODE -ne 0) { throw 'x86 ISA probe failed' }
Write-Host "host_isa=$isaOutput"
if ($Isa -eq 'avx' -and $isaOutput -notmatch 'avx_usable=1') {
  throw 'the Windows runner cannot safely execute AVX code'
}
if ($Isa -eq 'avx512' -and $isaOutput -notmatch 'avx512_usable=1') {
Append-Summary @"
## Windows 2025 / AVX-512

AVX-512 unavailable on this runner. The runtime probe reported:

$isaOutput

AVX-512 unavailable on this runner; optimization remains performance-untested.
No AVX-512 build or benchmark was executed.
"@
  exit 0
}

if ($Isa -eq 'avx') { $archFlag = '/arch:AVX' }
else { $archFlag = '/arch:AVX512' }

$env:CC = $clang
$env:CXX = $clang
$env:CMAKE_GENERATOR = 'Ninja'
Remove-Item Env:CMAKE_GENERATOR_PLATFORM -ErrorAction SilentlyContinue
Remove-Item Env:CMAKE_GENERATOR_TOOLSET -ErrorAction SilentlyContinue
$env:DUCC0_OPTIMIZATION = 'native-strip'
$env:CMAKE_ARGS = "-DDUCC0_ARCH_FLAGS=$archFlag"
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'

if ($Isa -eq 'avx') {
  $expected = 'DUCC_EXPECT_AVX'
  $case = 'nufft'
  $apiRepetitions = 5
  $testPath = 'python/test'
}
else {
  $expected = 'DUCC_EXPECT_AVX512'
  $case = 'sht'
  $apiRepetitions = 6
  $testPath = 'python/test'
}

$baselineVenv = Join-Path $workRoot 'baseline-venv'
Invoke-Checked $python @('-m', 'venv', $baselineVenv)
$baselinePython = Join-Path $baselineVenv 'Scripts\python.exe'
Invoke-Checked $baselinePython @('-m', 'pip', 'install', '--upgrade', 'pip')
Invoke-Checked $baselinePython @('-m', 'pip', 'install', '--upgrade',
  'pytest', 'numpy', 'scipy', 'scikit-build-core', 'nanobind', 'pybind11')
$requirements = Join-Path $workRoot 'requirements.txt'
& $baselinePython '-m' 'pip' 'freeze' | Out-File -Encoding utf8 $requirements
if ($LASTEXITCODE -ne 0) { throw 'could not record baseline dependency versions' }

$candidateConfigs = @()
if ($Isa -eq 'avx') {
  $candidateVenv = Join-Path $workRoot 'candidate-venv'
  Invoke-Checked $python @('-m', 'venv', $candidateVenv)
  $candidatePython = Join-Path $candidateVenv 'Scripts\python.exe'
  Invoke-Checked $candidatePython @('-m', 'pip', 'install', '--upgrade',
    '--requirement', $requirements)
  $candidateConfigs += [pscustomobject]@{
    Name = 'candidate'; Variant = 0; CFlags = ''; Python = $candidatePython
    Build = (Join-Path $workRoot 'candidate-build')
  }
}
else {
  $variantConfigs = @(
    @{ Name = 'candidate-a'; Variant = 1; CFlags = '/DDUCC0_SHT_AVX512_REDUCTION_VARIANT=1' },
    @{ Name = 'candidate-b'; Variant = 2; CFlags = '/DDUCC0_SHT_AVX512_REDUCTION_VARIANT=2' }
  )
  foreach ($variantConfig in $variantConfigs) {
    $candidateVenv = Join-Path $workRoot "$($variantConfig.Name)-venv"
    Invoke-Checked $python @('-m', 'venv', $candidateVenv)
    $candidatePython = Join-Path $candidateVenv 'Scripts\python.exe'
    Invoke-Checked $candidatePython @('-m', 'pip', 'install', '--upgrade',
      '--requirement', $requirements)
    $candidateConfigs += [pscustomobject]@{
      Name = $variantConfig.Name; Variant = $variantConfig.Variant
      CFlags = $variantConfig.CFlags; Python = $candidatePython
      Build = (Join-Path $workRoot "$($variantConfig.Name)-build")
    }
  }
}

$baselineBuild = Join-Path $workRoot 'baseline-build'
Install-Ducc -PythonExe $baselinePython -SourceDir $baselineSource -BuildDir $baselineBuild -Label 'baseline'
foreach ($candidateConfig in $candidateConfigs) {
  Install-Ducc -PythonExe $candidateConfig.Python -SourceDir $root -BuildDir $candidateConfig.Build -Label $candidateConfig.Name -DuccCFlags $candidateConfig.CFlags
}

$probeBaseline = Compile-SimdProbe -SourceDir $baselineSource -ExpectedMacro $expected -Label 'baseline' -ArchFlag $archFlag -Clang $clang -Root $root -OutDir $workRoot
$probeCandidates = @()
foreach ($candidateConfig in $candidateConfigs) {
  $probeCandidates += Compile-SimdProbe -SourceDir $root -ExpectedMacro $expected -Label $candidateConfig.Name -ArchFlag $archFlag -Clang $clang -Root $root -OutDir $workRoot
}
$shtProbeBaseline = $null
$shtProbeCandidates = @()
if ($Isa -eq 'avx512') {
  $shtProbeBaseline = Compile-ShtProbe -SourceDir $baselineSource -Label 'baseline-generic' -Clang $clang -Root $root -OutDir $workRoot -ArchFlag $archFlag -Variant 0
  foreach ($candidateConfig in $candidateConfigs) {
    $shtProbeCandidates += Compile-ShtProbe -SourceDir $root -Label $candidateConfig.Name -Clang $clang -Root $root -OutDir $workRoot -ArchFlag $archFlag -Variant $candidateConfig.Variant
  }
}

Write-Host "Running $Isa correctness tests for baseline"
Push-Location $baselineSource
try { Invoke-Checked $baselinePython @('-m', 'pytest', $testPath) }
finally { Pop-Location }
foreach ($candidateConfig in $candidateConfigs) {
  Write-Host "Running $Isa correctness tests for $($candidateConfig.Name)"
  Push-Location $root
  try { Invoke-Checked $candidateConfig.Python @('-m', 'pytest', $testPath) }
  finally { Pop-Location }
}

$avx512VerifiedLabels = @()
foreach ($candidateConfig in $candidateConfigs) {
  Compare-PythonCase -BaselinePython $baselinePython -CandidatePython $candidateConfig.Python -Root $root -Case $case -WorkRoot $workRoot -Label $candidateConfig.Name -Repetitions $apiRepetitions
  if ($Isa -eq 'avx512' -and $script:LastComparisonVerified) {
    $avx512VerifiedLabels += $candidateConfig.Name
  }
}
if ($Isa -eq 'avx512' -and $avx512VerifiedLabels.Count -eq 0) {
  throw 'neither AVX-512 SHT candidate showed a repeatable improvement over the exact-baseline generic reduction'
}

if ($Isa -eq 'avx') {
  $candidatePython = $candidateConfigs[0].Python
  foreach ($mode in @('sse3', 'sse2')) {
    $hsumBaseline = Compile-HsumBenchmark -SourceDir $baselineSource -Label 'baseline' -Clang $clang -Root $root -OutDir $workRoot -Mode $mode
    $hsumCandidate = Compile-HsumBenchmark -SourceDir $root -Label 'candidate' -Clang $clang -Root $root -OutDir $workRoot -Mode $mode
    $hsumBaseRuns = @()
    $hsumCandidateRuns = @()
    for ($round=0; $round -lt 4; ++$round) {
      if (($round % 2) -eq 0) { $order = @('candidate', 'baseline') }
      else { $order = @('baseline', 'candidate') }
      foreach ($which in $order) {
        $runDir = Join-Path $workRoot "hsum-$mode-$which-$round"
        New-Item -ItemType Directory -Force -Path $runDir | Out-Null
        if ($which -eq 'baseline') {
          Invoke-Checked $hsumBaseline @('1000000', '2', '5', $runDir)
          $hsumBaseRuns += $runDir
        }
        else {
          Invoke-Checked $hsumCandidate @('1000000', '2', '5', $runDir)
          $hsumCandidateRuns += $runDir
        }
      }
    }
    if ($mode -eq 'sse3') { $targetLabel = 'SSE3/no-AVX' }
    else { $targetLabel = 'SSE2-only/no-AVX' }
    $hsumSummary = Join-Path $workRoot "hsum-$mode-summary.md"
    $hsumArgs = @((Join-Path $root 'ci\simd_validation.py'), 'compare-hsum',
      '--summary', $hsumSummary, '--label', $targetLabel)
    foreach ($runDir in $hsumBaseRuns) { $hsumArgs += @('--baseline', $runDir) }
    foreach ($runDir in $hsumCandidateRuns) { $hsumArgs += @('--candidate', $runDir) }
    & $candidatePython @hsumArgs
    $hsumExit = $LASTEXITCODE
    if (Test-Path $hsumSummary) { Get-Content $hsumSummary | Add-Content $env:GITHUB_STEP_SUMMARY }
    if ($hsumExit -ne 0) { throw "focused $targetLabel comparison failed ($hsumExit)" }
  }
}

$candidateFlagsSummary = (($candidateConfigs | ForEach-Object {
  "$($_.Name): $($_.CFlags)"
}) -join '<br>')

$summaryHeader = @"
## Windows 2025 / $($Isa.ToUpper())

| Configuration | Value |
|---|---|
| Baseline SHA | $baselineSha |
| Candidate SHA | $candidateSha |
| Compiler | $clang |
| Architecture target | $archFlag |
| DUCC0 optimization | $env:DUCC0_OPTIMIZATION |
| CMAKE_ARGS | $env:CMAKE_ARGS |
| Candidate extra compile flags | $candidateFlagsSummary |
| API nthreads | 1 |
| API warmups / repetitions | 2 / $apiRepetitions |
| API paired rounds | 4, balanced AB/BA |
| hsum nthreads | 1 (single-process benchmark) |
| hsum warmups / repetitions | 2 / 5 |
| hsum paired rounds | 4, balanced AB/BA |
| Host ISA probe | $isaOutput |
| DUCC SIMD probe (baseline) | $probeBaseline |
| DUCC SIMD probe (candidate configurations) | $($probeCandidates -join '<br>') |
| SHT AVX-512 probe (baseline) | $shtProbeBaseline |
| SHT AVX-512 probes (candidate configurations) | $($shtProbeCandidates -join '<br>') |
| Candidate SHT variants | generic baseline; candidate A = `_mm512_reduce_add_pd`; candidate B = 512-to-256 fold plus existing fused AVX reduction |
| AVX-512 candidates passing the improvement gate | $($avx512VerifiedLabels -join ', ') |

"@
Append-Summary $summaryHeader
if ($Isa -eq 'avx') {
  Append-Summary "The AVX package result above is the general correctness/regression case. The separate SSE3/no-AVX table verifies that candidate `_mm_hadd_ps` remains selected; the SSE2-only table guards the generic fallback."
}
else {
  Append-Summary "The SHT tables above compare both AVX-512 candidates with the exact PR #68 baseline generic reduction. Candidate A uses `_mm512_reduce_add_pd`; candidate B folds 512-bit lanes to 256 bits and reuses the existing fused AVX reduction. The job runs only after the CPUID + XCR0 target-feature probe passes."
}
