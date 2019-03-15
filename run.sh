#!/bin/bash
python main.py --mode valid --gpu 1 --label intent
python main.py --mode ensemble --gpu 1 --label intent
python main.py --mode test --gpu 1 --label intent