# askpass.ps1 — Windows popup: request a per-app LiteLLM API key.
#
# CONSTITUTIONAL KEY LAW (AGENTS.md): one key per app, never the master key.
# This script pops a secure dialog, takes the key, and writes it ONLY to the
# app-secure store (never to any repo, never to .env, never to the master slot).
#
# Usage (Windows Shell / cmd / powershell):
#   powershell -ExecutionPolicy Bypass -File askpass.ps1 -App pool-model-zoo
#   powershell -ExecutionPolicy Bypass -File askpass.ps1 -App pool-sweep -Show
#
# The store lives OUTSIDE any repo: C:\Users\myste\.config\opencode\keys\<app>.key
# Each app gets exactly one key; the key limits the models that app may use.

param(
  [Parameter(Mandatory=$true)][string]$App,
  [switch]$Show  # print the key that would be used (for debugging the app name)
)

$ErrorActionPreference = "Stop"
$keysDir = Join-Path $env:USERPROFILE ".config\opencode\keys"
$keyFile = Join-Path $keysDir "$App.key"

New-Item -ItemType Directory -Path $keysDir -Force | Out-Null

if (Test-Path $keyFile) {
  $existing = (Get-Content $keyFile -Raw).Trim()
  if ($Show) { Write-Output "key file present: $keyFile (len $($existing.Length))" ; exit 0 }
  # Ask again ONLY if --force; otherwise confirm reuse.
  Write-Host "Existing key for '$App' present at $keyFile (len $($existing.Length))." -ForegroundColor Cyan
  $reuse = (Read-Host "Reuse it (R), or replace (N)?").ToUpper()
  if ($reuse -ne "N") {
    Write-Output "KEY_FILE=$keyFile`nKEY_SET=existing"
    exit 0
  }
}

# ---- secure popup (WinForms) -------------------------------------------------
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$form = New-Object System.Windows.Forms.Form
$form.Text = "LiteLLM API key -> $App"
$form.Size = New-Object System.Drawing.Size(560, 180)
$form.StartPosition = "CenterScreen"
$form.TopMost = $true
$form.FormBorderStyle = "FixedDialog"
$form.MaximizeBox = $false
$form.MinimizeBox = $false

$lbl = New-Object System.Windows.Forms.Label
$lbl.Text = "Enter the LiteLLM API key for app '$App'.`nThis key is registered per-app (never the master key) and limits the models this app may use."
$lbl.AutoSize = $true
$lbl.Location = New-Object System.Drawing.Point(20, 18)
$form.Controls.Add($lbl)

$txt = New-Object System.Windows.Forms.TextBox
$txt.Width = 505
$txt.Location = New-Object System.Drawing.Point(20, 60)
$txt.UseSystemPasswordChar = $true
$form.Controls.Add($txt)

$ok = New-Object System.Windows.Forms.Button
$ok.Text = "Save key"
$ok.DialogResult = "OK"
$ok.Location = New-Object System.Drawing.Point(150, 100)
$form.Controls.Add($ok)

$cancel = New-Object System.Windows.Forms.Button
$cancel.Text = "Cancel"
$cancel.DialogResult = "Cancel"
$cancel.Location = New-Object System.Drawing.Point(300, 100)
$form.Controls.Add($cancel)

$form.AcceptButton = $ok
$form.CancelButton = $cancel

$result = $form.ShowDialog()
if ($result -ne "OK") { Write-Error "Cancelled — no key stored for $App"; exit 1 }

$key = $txt.Text.Trim()
if ($key.Length -lt 8) { Write-Error "Key too short/empty — refusing to store."; exit 1 }

# atomic write (temp + move) so a crash never leaves a partial key
$tmp = "$keyFile.tmp"
[System.IO.File]::WriteAllText($tmp, $key)
Move-Item -Force $tmp $keyFile

Write-Output "KEY_FILE=$keyFile"
Write-Output "KEY_SET=new"
Write-Output "Key stored for app '$App' (len $($key.Length)). App models are limited by the key's registered model list per AGENTS.md law."