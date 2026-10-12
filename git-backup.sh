#!/bin/bash
cd ~/printer_data/config
git add -A
git commit -m "Autocommit after print: $(date '+%Y-%m-%d %H:%M')"
git push origin main
