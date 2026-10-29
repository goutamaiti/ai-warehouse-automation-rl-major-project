# Runs scripts/run_seed_study.py while preventing Windows from idle-sleeping.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_seed_study_awake.ps1
#
# SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) holds only while
# this process lives and is released automatically when it exits - no power
# setting is changed. A closed laptop lid still sleeps the machine; leave it open.
param([string[]]$StudyArgs = @())

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
[void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
Write-Host "Keeping the computer awake until the seed study finishes."

try {
    python scripts\run_seed_study.py @StudyArgs
    $code = $LASTEXITCODE
} finally {
    [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS)
    Write-Host "Seed study exited; sleep allowed again."
}
exit $code
