#!/bin/bash
python3 incremental_pred.py --data_path data/seen.all.strided.1024.pdnc.4096.right0.json --exp full.incremental.seen.s1024.4096.r0
# python3 incremental_pred.py --data_path data/test.all.strided.1024.pdnc.4096.right0.json --exp full.incremental.test.s1024.4096.r0
# python3 incremental_pred.py --data_path data/unseen.all.strided.1024.pdnc.4096.right0.json --exp incremental.unseen.s1024.4096.r0
# python3 incremental_pred.py --data_path data/test.all.strided.1024.pdnc.4096.right0.json --exp no_incremental.test.s1024.4096.r0 --stop_incremental
