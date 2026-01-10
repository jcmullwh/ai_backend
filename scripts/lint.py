import logging
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# List of allowed commands for validation
ALLOWED_COMMANDS = [["ruff", "format", "."], ["ruff", "check", "."], ["mypy", "src/"]]


def run_command(command):
    if command not in ALLOWED_COMMANDS:
        logging.error(f"Attempted to run an untrusted command: {' '.join(command)}")
        sys.exit(1)

    executable = shutil.which(command[0])
    if executable is None:
        candidate = Path(sys.executable).with_name(f"{command[0]}.exe")
        if candidate.exists():
            executable = str(candidate)
        else:
            logging.error(f"Required command '{command[0]}' is not installed or not on PATH.")
            sys.exit(1)

    full_command = [executable, *command[1:]]

    try:
        subprocess.run(full_command, check=True)  # noqa: S603
    except subprocess.CalledProcessError as e:
        logging.error(f"Command '{' '.join(command)}' failed with exit code {e.returncode}")
        sys.exit(e.returncode)


# Commands to run
commands = [["ruff", "format", "."], ["ruff", "check", "."], ["mypy", "src/"]]

# Execute each command
for command in commands:
    logging.info(f"Running: {' '.join(command)}")
    run_command(command)
