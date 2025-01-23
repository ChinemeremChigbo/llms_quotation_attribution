#!/bin/bash
python main_name_cloze.py --data_path data/pdnc_source/ --num_examples 100
python main_name_cloze.py --data_path data/test_pdnc_source/ --num_examples 100
python main_name_cloze.py --data_path data/unseen_source/ --num_examples 100
python main_speaker_cloze.py --data_path data/pdnc_source/ --num_quotes 100
python main_speaker_cloze.py --data_path data/test_pdnc_source/ --num_quotes 100 
python main_speaker_cloze.py --data_path data/unseen_source/ --num_quotes 100 
