// Generic in-place setup wizard for a Buzzcaf Media app (scripts/app_installer, see README.md).
// The app is installed where its folder already is: this wizard (re)builds the app's
// environment with installer\setup_env.ps1, makes the Desktop and Start Menu shortcuts,
// registers the app in Apps & features and writes installer\installed.json, the marker
// SmartCleaner reads to leave this app's environments alone.
// Everything app-specific comes from AppInfo.cs, generated from installer\app.json by
// build_installer.ps1. C# 5 (the .NET Framework csc), so no interpolation or ?. here.
using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using Microsoft.Win32;

namespace BuzzcafAppSetup
{
    static class Program
    {
        [DllImport("user32.dll")]
        private static extern bool SetProcessDPIAware();

        [STAThread]
        static void Main(string[] args)
        {
            try { SetProcessDPIAware(); } catch { }
            // /quiet: no dialogs, and the app is not started afterwards unless /launch is given too
            // (a script installing several apps would otherwise open them all). Git Bash rewrites a
            // bare /quiet into a path, so the --quiet / --launch spellings work as well.
            bool quiet = false, launch = false;
            foreach (string a in args)
            {
                if (a == "/quiet" || a == "--quiet") quiet = true;
                if (a == "/launch" || a == "--launch") launch = true;
            }
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new SetupForm(quiet, launch));
        }
    }

    public static class Paths
    {
        // installer\<Name> Setup.exe -> the app's root is the folder above installer\.
        public static string Root()
        {
            string here = AppDomain.CurrentDomain.BaseDirectory.TrimEnd('\\');
            if (Path.GetFileName(here).Equals("installer", StringComparison.OrdinalIgnoreCase))
                return Path.GetDirectoryName(here);
            return here;
        }

        public static string Expand(string value, string root)
        {
            return Environment.ExpandEnvironmentVariables(value.Replace("{root}", root));
        }

        public static string StartMenuDir()
        {
            return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Programs), AppInfo.Publisher);
        }

        public static string DesktopLink()
        {
            return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), AppInfo.Name + ".lnk");
        }

        public static string StartMenuLink()
        {
            return Path.Combine(StartMenuDir(), AppInfo.Name + ".lnk");
        }

        public static string UninstallLink()
        {
            return Path.Combine(StartMenuDir(), "Uninstall " + AppInfo.Name + ".lnk");
        }

        public static string RegistryKey()
        {
            return @"Software\Microsoft\Windows\CurrentVersion\Uninstall\Buzzcaf." + AppInfo.Id;
        }

        public static string Marker(string root)
        {
            return Path.Combine(root, "installer", "installed.json");
        }
    }

    public class SetupForm : Form
    {
        private readonly string root;
        private readonly bool quiet;
        private CheckBox chkEnv, chkDesktop, chkStartMenu, chkLaunch;
        private Button btnInstall, btnClose;
        private TextBox txtLog;
        private Label lblStatus;
        private ProgressBar bar;

        private static readonly Color Bg = Color.FromArgb(18, 20, 29);
        private static readonly Color Header = Color.FromArgb(14, 16, 23);
        private static readonly Color Accent = Color.FromArgb(251, 146, 60);
        private static readonly Color Ink = Color.FromArgb(241, 245, 249);
        private static readonly Color Muted = Color.FromArgb(148, 163, 184);

        public SetupForm(bool quiet, bool launch)
        {
            this.root = Paths.Root();
            this.quiet = quiet;
            Build();
            if (quiet) chkLaunch.Checked = launch;
            if (quiet) this.Shown += delegate { BeginInstall(); };
        }

        private void Build()
        {
            Text = AppInfo.Name + " Setup";
            Size = new Size(720, 600);
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = false;
            StartPosition = FormStartPosition.CenterScreen;
            BackColor = Bg;
            ForeColor = Ink;
            Font = new Font("Segoe UI", 9.5f);
            string icon = Path.Combine(root, AppInfo.Icon);
            if (AppInfo.Icon.Length > 0 && File.Exists(icon)) { try { Icon = new Icon(icon); } catch { } }

            Panel head = new Panel { Dock = DockStyle.Top, Height = 78, BackColor = Header };
            head.Controls.Add(new Label { Text = AppInfo.Name, Font = new Font("Segoe UI", 15f, FontStyle.Bold), ForeColor = Color.White, Location = new Point(22, 12), AutoSize = true });
            head.Controls.Add(new Label { Text = AppInfo.Description, ForeColor = Accent, Location = new Point(24, 46), AutoSize = true });
            Controls.Add(head);

            Controls.Add(new Label { Text = "Installs in place, in this folder:", ForeColor = Muted, Location = new Point(22, 92), AutoSize = true });
            Controls.Add(new TextBox { Text = root, ReadOnly = true, Location = new Point(22, 114), Width = 660, BackColor = Header, ForeColor = Ink, BorderStyle = BorderStyle.FixedSingle });

            chkEnv = Check("Set up or repair the environment (" + AppInfo.EnvSummary + ")", 150, true);
            chkDesktop = Check("Desktop shortcut", 176, true);
            chkStartMenu = Check("Start Menu shortcut (" + AppInfo.Publisher + ")", 202, true);
            chkLaunch = Check("Start " + AppInfo.Name + " when setup finishes", 228, true);

            txtLog = new TextBox
            {
                Multiline = true, ReadOnly = true, ScrollBars = ScrollBars.Vertical, WordWrap = false,
                Location = new Point(22, 262), Size = new Size(660, 210),
                BackColor = Header, ForeColor = Ink, Font = new Font("Consolas", 8.5f), BorderStyle = BorderStyle.FixedSingle
            };
            Controls.Add(txtLog);

            bar = new ProgressBar { Location = new Point(22, 482), Size = new Size(660, 8), Style = ProgressBarStyle.Continuous };
            Controls.Add(bar);
            lblStatus = new Label { Text = "Ready.", ForeColor = Muted, Location = new Point(22, 500), Size = new Size(440, 40) };
            Controls.Add(lblStatus);

            btnInstall = new Button { Text = "Install", Location = new Point(478, 506), Size = new Size(100, 32), FlatStyle = FlatStyle.Flat, BackColor = Accent, ForeColor = Color.Black };
            btnInstall.Click += delegate { BeginInstall(); };
            Controls.Add(btnInstall);
            btnClose = new Button { Text = "Cancel", Location = new Point(584, 506), Size = new Size(98, 32), FlatStyle = FlatStyle.Flat, ForeColor = Ink };
            btnClose.Click += delegate { Close(); };
            Controls.Add(btnClose);
            AcceptButton = btnInstall;
        }

        private CheckBox Check(string text, int y, bool on)
        {
            CheckBox c = new CheckBox { Text = text, Checked = on, Location = new Point(22, y), AutoSize = true, ForeColor = Ink };
            Controls.Add(c);
            return c;
        }

        private void Log(string line)
        {
            if (line == null) return;
            if (InvokeRequired) { BeginInvoke(new Action<string>(Log), line); return; }
            txtLog.AppendText(line + Environment.NewLine);
        }

        private void Status(string text, int percent)
        {
            if (InvokeRequired) { BeginInvoke(new Action<string, int>(Status), text, percent); return; }
            lblStatus.Text = text;
            bar.Value = Math.Max(0, Math.Min(100, percent));
        }

        private void BeginInstall()
        {
            btnInstall.Enabled = false;
            btnClose.Enabled = false;
            chkEnv.Enabled = chkDesktop.Enabled = chkStartMenu.Enabled = chkLaunch.Enabled = false;
            bool env = chkEnv.Checked, desk = chkDesktop.Checked, menu = chkStartMenu.Checked, launch = chkLaunch.Checked;
            Thread t = new Thread(delegate () { Install(env, desk, menu, launch); });
            t.IsBackground = true;
            t.Start();
        }

        private void Install(bool env, bool desk, bool menu, bool launch)
        {
            string failure = null;
            try
            {
                if (env)
                {
                    Status("Setting up the environment. The first run downloads dependencies and can take a while.", 5);
                    int code = RunSetupScript();
                    if (code != 0) failure = "The environment setup stopped with exit code " + code + ". The log above says where; fix it and run Setup again (it picks up where it stopped).";
                }
                if (failure == null)
                {
                    Status("Creating shortcuts...", 85);
                    string uninstaller = Path.Combine(root, "installer", "Uninstall " + AppInfo.Name + ".exe");
                    string target = Paths.Expand(AppInfo.LaunchTarget, root);
                    string args = Paths.Expand(AppInfo.LaunchArgs, root);
                    string icon = Path.Combine(root, AppInfo.Icon);
                    if (desk) { Shortcut(Paths.DesktopLink(), target, args, icon); Log("Desktop shortcut: " + Paths.DesktopLink()); }
                    if (menu)
                    {
                        Directory.CreateDirectory(Paths.StartMenuDir());
                        Shortcut(Paths.StartMenuLink(), target, args, icon);
                        if (File.Exists(uninstaller)) Shortcut(Paths.UninstallLink(), uninstaller, "", icon);
                        Log("Start Menu shortcut: " + Paths.StartMenuLink());
                    }
                    Register(uninstaller, icon);
                    WriteMarker();
                    Log("Registered in Apps & features as \"" + AppInfo.Name + "\".");
                    Status("Done. " + AppInfo.Name + " is installed.", 100);
                    if (launch) { try { Process.Start(new ProcessStartInfo(target, args) { WorkingDirectory = root, UseShellExecute = true }); } catch (Exception e) { Log("Could not start it: " + e.Message); } }
                }
            }
            catch (Exception e) { failure = e.Message; }

            BeginInvoke(new Action(delegate
            {
                btnClose.Enabled = true;
                btnClose.Text = "Close";
                if (failure != null)
                {
                    Status("Setup did not finish.", bar.Value);
                    Log("FAILED: " + failure);
                    btnInstall.Enabled = true;
                    btnInstall.Text = "Retry";
                    if (!quiet) MessageBox.Show(this, failure, AppInfo.Name + " Setup", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                if (quiet) { Environment.ExitCode = failure == null ? 0 : 1; Close(); }
            }));
        }

        private int RunSetupScript()
        {
            string script = Path.Combine(root, AppInfo.SetupScript);
            if (AppInfo.SetupScript.Length == 0 || !File.Exists(script)) { Log("No environment script; nothing to set up."); return 0; }
            ProcessStartInfo psi = new ProcessStartInfo("powershell.exe",
                "-NoProfile -ExecutionPolicy Bypass -File \"" + script + "\"")
            {
                WorkingDirectory = root,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8
            };
            psi.EnvironmentVariables["BUZZ_SETUP_GUI"] = "1";
            Process p = new Process { StartInfo = psi };
            p.OutputDataReceived += delegate (object s, DataReceivedEventArgs e) { Progress(e.Data); };
            p.ErrorDataReceived += delegate (object s, DataReceivedEventArgs e) { Log(e.Data); };
            p.Start();
            p.BeginOutputReadLine();
            p.BeginErrorReadLine();
            p.WaitForExit();
            return p.ExitCode;
        }

        // setup_common.ps1 prints "##STEP <n>/<total> <text>" before each step.
        private void Progress(string line)
        {
            if (line == null) return;
            if (line.StartsWith("##STEP "))
            {
                string rest = line.Substring(7);
                int sp = rest.IndexOf(' ');
                string[] nt = (sp > 0 ? rest.Substring(0, sp) : rest).Split('/');
                int n, total;
                if (nt.Length == 2 && int.TryParse(nt[0], out n) && int.TryParse(nt[1], out total) && total > 0)
                    Status(sp > 0 ? rest.Substring(sp + 1) : "", 5 + 78 * (n - 1) / total);
                Log("== " + (sp > 0 ? rest.Substring(sp + 1) : rest));
                return;
            }
            Log(line);
        }

        private static void Shortcut(string lnk, string target, string args, string icon)
        {
            Type shellType = Type.GetTypeFromProgID("WScript.Shell");
            object shell = Activator.CreateInstance(shellType);
            object sc = shellType.InvokeMember("CreateShortcut", BindingFlags.InvokeMethod, null, shell, new object[] { lnk });
            Type t = sc.GetType();
            t.InvokeMember("TargetPath", BindingFlags.SetProperty, null, sc, new object[] { target });
            t.InvokeMember("Arguments", BindingFlags.SetProperty, null, sc, new object[] { args });
            t.InvokeMember("WorkingDirectory", BindingFlags.SetProperty, null, sc, new object[] { Paths.Root() });
            t.InvokeMember("Description", BindingFlags.SetProperty, null, sc, new object[] { AppInfo.Name + " - " + AppInfo.Description });
            if (File.Exists(icon)) t.InvokeMember("IconLocation", BindingFlags.SetProperty, null, sc, new object[] { icon + ",0" });
            t.InvokeMember("Save", BindingFlags.InvokeMethod, null, sc, null);
        }

        private void Register(string uninstaller, string icon)
        {
            using (RegistryKey k = Registry.CurrentUser.CreateSubKey(Paths.RegistryKey()))
            {
                k.SetValue("DisplayName", AppInfo.Name);
                k.SetValue("DisplayVersion", AppInfo.Version);
                k.SetValue("Publisher", AppInfo.Publisher);
                k.SetValue("InstallLocation", root);
                k.SetValue("DisplayIcon", File.Exists(icon) ? icon : uninstaller);
                k.SetValue("UninstallString", "\"" + uninstaller + "\"");
                k.SetValue("ModifyPath", "\"" + Path.Combine(root, "installer", AppInfo.Name + " Setup.exe") + "\"");
                k.SetValue("NoRepair", 1, RegistryValueKind.DWord);
                k.SetValue("EstimatedSize", AppInfo.EstimatedMB * 1024, RegistryValueKind.DWord);
                k.SetValue("InstallDate", DateTime.Now.ToString("yyyyMMdd"));
            }
        }

        private void WriteMarker()
        {
            string json = "{\n  \"app\": \"" + AppInfo.Id + "\",\n  \"name\": \"" + AppInfo.Name + "\",\n  \"version\": \"" + AppInfo.Version +
                          "\",\n  \"installed\": \"" + DateTime.Now.ToString("yyyy-MM-dd HH:mm") +
                          "\",\n  \"note\": \"Written by the setup wizard. SmartCleaner never offers this app's environments (venv, node_modules, target) while this file exists.\"\n}\n";
            File.WriteAllText(Paths.Marker(root), json, new UTF8Encoding(false));
        }
    }
}
