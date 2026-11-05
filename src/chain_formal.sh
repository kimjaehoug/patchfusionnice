#!/usr/bin/env bash
cd "$(dirname "$0")"
until grep -q "queue finished" ../logs/run_eicu4.stdout; do sleep 30; done
~/.venv/bin/python run_queue.py jobs_formal.txt --parallel 4 --gpu 0 > ../logs/run_formal.stdout 2>&1
