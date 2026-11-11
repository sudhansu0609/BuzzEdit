// Generic uninstaller for an app installed in place by Setup.cs (scripts/app_installer).
// It undoes what Setup did -- shortcuts, the Apps & features entry and installer\installed.json --
// and never deletes the app's folder, its data or its environments: those are the creator's.
using System;
using System.IO;
using System.Windows.Forms;
using Microsoft.Win32;

namespace BuzzcafAppSetup
{
    static class UninstallProgram
    {
        [STAThread]
        static void Main(string[] args)
        {
            bool quiet = false;
            foreach (string a in args) if (a == "/quiet" || a == "--quiet") quiet = true;
            Application.EnableVisualStyles();
            string root = Paths.Root();
            if (!quiet && MessageBox.Show(
                    "Remove " + AppInfo.Name + " from Windows?\n\n" +
                    "This removes its shortcuts and its Apps & features entry. The folder\n" + root +
                    "\nand everything in it (your data, models and environments) stays as it is.",
                    "Uninstall " + AppInfo.Name, MessageBoxButtons.OKCancel, MessageBoxIcon.Question) != DialogResult.OK)
                return;

            foreach (string lnk in new[] { Paths.DesktopLink(), Paths.StartMenuLink(), Paths.UninstallLink() })
            {
                try { if (File.Exists(lnk)) File.Delete(lnk); } catch { }
            }
            try
            {
                string dir = Paths.StartMenuDir();
                if (Directory.Exists(dir) && Directory.GetFileSystemEntries(dir).Length == 0) Directory.Delete(dir);
            }
            catch { }
            try { Registry.CurrentUser.DeleteSubKeyTree(Paths.RegistryKey(), false); } catch { }
            try { if (File.Exists(Paths.Marker(root))) File.Delete(Paths.Marker(root)); } catch { }

            if (!quiet)
                MessageBox.Show(AppInfo.Name + " was removed from Windows.\n\nRun installer\\" + AppInfo.Name +
                                " Setup.exe to install it again.", "Uninstall " + AppInfo.Name,
                                MessageBoxButtons.OK, MessageBoxIcon.Information);
        }
    }
}
