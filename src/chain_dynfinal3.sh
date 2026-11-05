#!/usr/bin/env bash
# start the three-regime queue when the two-regime formal queue has finished
cd "$(dirname "$0")"
until grep -q "queue finished" ../logs/run_dynfinal.stdout; do sleep 60; done
~/.venv/bin/python run_queue.py jobs_dynfinal3.txt --parallel 3 --gpu 0 > ../logs/run_dynfinal3.stdout 2>&1
