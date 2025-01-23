#!/bin/bash
git clone git@github.com:Priya22/project-dialogism-novel-corpus.git
python main_contamination.py --pdnc_path project-dialogism-novel-corpus/data/
python main_contamination.py --pdnc_path data/Darkcorners/
