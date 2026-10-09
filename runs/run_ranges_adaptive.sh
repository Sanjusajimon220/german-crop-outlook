#!/bin/bash
cd /d/vista
until grep -q "forward in-season done" data/processed/forward_inseason.log; do sleep 60; done
PYTHONIOENCODING=utf-8 "/c/Users/Asus/AppData/Local/Programs/Python/Python312/python.exe" -W ignore ranges_adaptive.py
