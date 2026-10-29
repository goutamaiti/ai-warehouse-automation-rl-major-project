# Run any project script while preventing Windows from idle-sleeping.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_awake.ps1 scripts\run_reward_studies.py
#   powershell -ExecutionPolicy Bypass -File scripts\run_awake.ps1 scripts\run_seed_study.py --seeds 3 4
#
# The no-sleep request holds only while this process lives and is released when
# it exits; no power setting is changed. A closed laptop lid still sleeps the
# machine - leave it open.
param([Parameter(Mandatory = $true)][string]$Script, [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest = @())

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
[void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
Write-Host "Keeping the computer awake until $Script finishes."
try {
    python $Script @Rest
    $code = $LASTEXITCODE
} finally {
    [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS)
    Write-Host "Finished; sleep allowed again."
}
exit $code
