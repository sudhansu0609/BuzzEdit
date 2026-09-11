' BuzzEdit silent launcher -- the double-click entry point.
'
' A .bat or .ps1 started directly always shows a console window for as long as it
' runs, which is the black cmd window that used to sit next to the app. WScript is
' a windowless host, so starting PowerShell from here with window style 0 means
' the launcher never draws anything: the Electron window is the only thing that
' appears.
'
' Everything it would have printed goes to logs\launcher.log.

Dim shell, fso, here, command
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
here = fso.GetParentFolderName(WScript.ScriptFullName)

command = "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass " & _
          "-WindowStyle Hidden -File """ & here & "\START_APP.ps1"""

' 0 = hidden window, False = do not block this script on it.
shell.Run command, 0, False
