#!/bin/bash
for env in /root/autodl-tmp/miniconda3/envs/*; do
    echo "Checking $env..."
    $env/bin/pip list | grep -i openai
done
