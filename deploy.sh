#!/bin/bash
# This script builds this project's source and binary distributions
# and uploads them to PyPI

# source configurations
source ".global.sh"

# check for prerequisites
if [[ $(which virtualenvwrapper.sh) == "" ]]; then
  logE "Could not find virtualenvwrapper utility"
  exit 4
fi
source $(which virtualenvwrapper.sh)

# ensure the virtual environment is active
if [[ "$VIRTUAL_ENV" == "" ]]; then
  if ! workon "$VIRTENV_NAME"; then
    logE "Failed to switch to virtual environment '$VIRTENV_NAME'"
    exit 2
  fi
fi

# dependencies
if [[ $(which twine) == "" ]]; then
  logE "Could not find twine"
  exit 4
fi

logI "Running unit tests"
$PYTHON_EXEC -m unittest
check_code=$?
if [ $check_code != 0 ]; then
  logE "Tests have failed"
  exit $check_code
fi
logI "All unit tests have passed"

logI "Building source and binary distribution packages"
$PYTHON_EXEC setup.py sdist bdist_wheel
check_code=$?
if [ $check_code != 0 ]; then
  logE "Failed to build distribution packages"
  exit $check_code
fi

logI "Running distribution checks"
twine check dist/*
check_code=$?
if [ $check_code != 0 ]; then
  logE "Distribution checks have failed"
  exit $check_code
fi
logI "All distribution checks have passed"

logI "Uploading distributions to PyPI"
twine upload dist/*
check_code=$?
if [ $check_code != 0 ]; then
  logE "Failed to upload distributions"
  exit $check_code
fi
logI "Distributions have been successfully uploaded"
