@echo off
ruff format --check .
ruff .
mypy src
