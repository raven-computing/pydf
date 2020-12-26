#!/bin/bash
# -----------------------------------------------------------------------------#
#                   Global configuration values and functions                  #
# -----------------------------------------------------------------------------#

# Virtual environment name
VIRTENV_NAME="pydf"

# Python executable
PYTHON_EXEC="python"

# PIP executable
PIP_EXEC="pip3"

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Terminal colors
USE_ANSI_COLORS=true
RED="\033[0;31m"
GREEN="\033[0;32m"
BLUE="\033[1;34m"
ORANGE="\033[1;33m"
NC="\033[0m"

# Print info level statement on stdout
function logI() {
  if [[ $USE_ANSI_COLORS == true ]]; then
    echo -e "[${BLUE}INFO${NC}] $*"
  else
    echo "[INFO] $*"
  fi
}

# Print warning level statement on stdout
function logW() {
  if [[ $USE_ANSI_COLORS == true ]]; then
    echo -e "[${ORANGE}WARN${NC}] $*"
  else
    echo "[WARN] $*"
  fi
}

# Print error level statement on stdout
function logE() {
  if [[ $USE_ANSI_COLORS == true ]]; then
    echo -e "[${RED}ERROR${NC}] $*"
  else
    echo "[ERROR] $*"
  fi
}
