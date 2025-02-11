#!/bin/bash
wget https://hanoverprod.z21.web.core.windows.net/med_llava/llava_med_image_urls.jsonl
mkdir pmc_articles
mkdir images
python script.py