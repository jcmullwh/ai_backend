@echo off
ruff format .
ruff --fix .
mypy src