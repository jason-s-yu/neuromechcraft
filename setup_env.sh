#!/usr/bin/bash

# check if pyenv is installed
if ! command -v pyenv &> /dev/null
then
  echo "pyenv not found. Please install pyenv."
else
  echo "pyenv is already installed"
fi

# check if pyenv-virtualenv is installed
if ! pyenv commands | grep -q 'virtualenv'
then
  echo "pyenv-virtualenv not found. Please install pyenv-virtualenv."
else
  echo "pyenv-virtualenv is already installed"
fi

# latest version of python with distutils is 3.11.11; required for minerl install
pyenv install 3.11.11 

pyenv virtualenv 3.11.11 neuromechcraft

pyenv activate neuromechcraft

python -m pip install --upgrade pip

pip install git+https://github.com/minerllabs/minerl
