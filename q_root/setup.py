from setuptools import setup
from setuptools.command.develop import develop
import os, json

class DevEnvSetup(develop):
    def run(self):
        root_path = os.path.abspath(os.path.dirname(__file__))

        env_path = os.path.join(root_path, '.env')
        with open(env_path, 'w') as f:
            f.write(f"PYTHONPATH={root_path.replace(os.sep, '/')}\n")
        print(f'[OK] .env created at {env_path}')

        vscode_dir = os.path.join(root_path, '.vscode')
        os.makedirs(vscode_dir, exist_ok=True)
        settings_path = os.path.join(vscode_dir, 'settings.json')
        settings = {
            "python.envFile": "${workspaceFolder}/.env",
            "python.analysis.extraPaths": ["."]
        }
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=4)
        print(f'[OK] .vscode/settings.json created at {settings_path}')

        self.patch_cursor_global_settings()

        super().run()

    def patch_cursor_global_settings(self):
        cursor_config_path = os.path.expandvars(r"%APPDATA%\Cursor\User\settings.json")
        try:
            with open(cursor_config_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing = {}

        changed = False
        if "python.envFile" not in existing:
            existing["python.envFile"] = "${workspaceFolder}/.env"
            changed = True
            
        if "python.analysis.extraPaths" not in existing:
            existing["python.analysis.extraPaths"] = [".", "./q_root", "${workspaceFolder}/q_root"]
            changed = True
        else:
            extra_paths = existing["python.analysis.extraPaths"]
            paths_to_add = ["./q_root", "${workspaceFolder}/q_root"]
            
            for path in paths_to_add:
                if path not in extra_paths:
                    extra_paths.append(path)
                    changed = True
            
            existing["python.analysis.extraPaths"] = extra_paths

        if changed:
            with open(cursor_config_path, 'w', encoding='utf-8') as f:
                json.dump(existing, f, indent=4)
            print(f"[OK] Cursor settings patched at: {cursor_config_path}")
        else:
            print("[OK] Cursor settings already configured.")

setup(
    cmdclass={
        'develop': DevEnvSetup
    }
)
