cd "$(dirname "$0")"

TARGET_VERSION=$(grep "__version__" bt/__version__.py | cut -d'"' -f2)

INSTALLED_VERSION=$(pip show bt 2>/dev/null | grep "^Version:" | cut -d' ' -f2)

echo "Target version: ${TARGET_VERSION}"
echo "Installed version: ${INSTALLED_VERSION}"

if [ "${INSTALLED_VERSION}" = "${TARGET_VERSION}" ]; then
    echo "[✓] bt ${INSTALLED_VERSION} already installed. Skipping reinstall."
else
    echo "[!] Installing bt ${TARGET_VERSION}..."
    pip uninstall -y bt
    pip install -e . --use-pep517
fi

mkdir -p ../.vscode

PYTHON_PATH=$(which python)

cat > ../.vscode/settings.json << EOF
{
  "[python]": {
    "editor.defaultFormatter": "ms-python.autopep8"
  },
  "python.formatting.provider": "none",
  "python.REPL.enableREPLSmartSend": false,
  "python.defaultInterpreterPath": "${PYTHON_PATH}",
  "python.analysis.extraPaths": ["${PWD}"]
}
EOF

echo "[✓] VS Code/Cursor settings configured."
echo "[OK] Development environment is ready!" 