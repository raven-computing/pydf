#!/bin/bash
# Setup script for the pydf project.
# This script will create and initialize a virtual environment
# for the project and set up the python path for you.

# source configurations
source ".global.sh"

function isinsubshell() {
  if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "false"
  else
    echo "true"
  fi
}

# determine whether this script is sourced or executed in a subshell
if [[ $(isinsubshell) == "true" ]]; then
  logE "Please run this script by sourcing it"
  exit 1
fi

# check for prerequisites
if [[ $(which virtualenvwrapper.sh) == "" ]]; then
  logE "Could not find virtualenvwrapper utility"
  return 4
fi
source $(which virtualenvwrapper.sh)
# Ensure the required executables can be found
if [[ $(which $PYTHON_EXEC) == "" ]]; then
  logE "Could not find requirement '$PYTHON_EXEC'"
  return 4
fi
if [[ $(which $PIP_EXEC) == "" ]]; then
  logE "Could not find requirement '$PIP_EXEC'"
  return 4
fi

# check if virtual environment home is set
if [[ "$WORKON_HOME" == "" ]]; then
  logE "Failed to access virtual environment home directory"
  logE "Please set 'WORKON_HOME' environment variable"
  return 2
fi

# check if virtual environment is already active
if [[ ! "$VIRTUAL_ENV" == "" ]]; then
  logW "A virtual environment is already active: $VIRTUAL_ENV"
  logW "Exit active virtual environment and try again"
  return 2
fi

# check if a virtual environment with the same name already exists
foundenv=$(lsvirtualenv |grep "$VIRTENV_NAME")
if [[ ! "$foundenv" == "" ]]; then
  # environment is already set up, so we simply switch to it
  if ! workon "$VIRTENV_NAME"; then
    logE "Failed to switch to virtual environment '$VIRTENV_NAME'"
    return 2
  fi
  # successfully switched to environment
  return 0
fi

# create the virtual environment
logI "Creating virtual environment '$VIRTENV_NAME'"
if ! mkvirtualenv "$VIRTENV_NAME"; then
  logE "Failed to create virtual environment"
  logE "'mkvirtualenv' returned non-zero exit status"
  return 2
fi

# ensure the created virtual environment is active
if [[ ! "$VIRTUAL_ENV" == "" ]]; then
  if ! workon "$VIRTENV_NAME"; then
    logE "Failed to switch to virtual environment '$VIRTENV_NAME'"
    return 2
  fi
fi

# ensure the current working directory is the project root
cd "$PROJECT_ROOT"

# install python dependencies
logI "Installing python dependencies"
$PIP_EXEC install -r requirements.txt

# add values from pythonpath file
if [ -f ".pythonpath" ]; then
  while read -r line; do
    # ignore comments
    if [[ ! "$line" == \#* ]]; then
      # substitude variables
      pypath=$(echo "$line" |envsubst)
      logI "Adding '$pypath' to python path"
      add2virtualenv "$pypath"
    fi
  done < ".pythonpath"
fi
logI "Setup completed"
