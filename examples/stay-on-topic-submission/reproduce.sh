#!/bin/bash
# show the commands being run
set -x

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y curl

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# source it
source $HOME/.local/bin/env

# install reqs
uv sync

# reproduce 3.1 (short reduced version of the experiment)
uv run python run_3.1.py --short

# reproduce 3.2 (commenting out because unable to run with current hardware)
# uv run python run_3.2.py
