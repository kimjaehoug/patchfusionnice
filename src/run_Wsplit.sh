#!/bin/bash
cd ~/patchfusionnice/src
~/.venv/bin/python run_queue.py jobs_Wsplit1.txt --parallel 3 --gpu 1 > ../logs/run_Wsplit1.stdout 2>&1 &
while kill -0 3601715 2>/dev/null || kill -0 3615785 2>/dev/null || kill -0 3621581 2>/dev/null; do sleep 30; done
~/.venv/bin/python run_queue.py jobs_Wsplit0.txt --parallel 3 --gpu 0 > ../logs/run_Wsplit0.stdout 2>&1
wait
