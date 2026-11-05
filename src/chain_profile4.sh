#!/usr/bin/env bash
cd "$(dirname "$0")"
until grep -q "queue finished" ../logs/run_profile3.stdout; do sleep 20; done
~/.venv/bin/python run_queue.py jobs_profile4.txt --parallel 3 --gpu 0 > ../logs/run_profile4.stdout 2>&1
