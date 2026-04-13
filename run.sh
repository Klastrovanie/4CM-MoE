#!/bin/bash
echo "=== TF-IDF Version ==="
python TfidfVectorizer-version/4cm-MoE.py

echo ""
echo "=== BERT Version ==="
python Transformer-version/4cm-MoE-BERT.py