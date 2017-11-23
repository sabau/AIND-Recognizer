#!/bin/bash

sed -i "s/'/\"/g" probabilities
sed -i "s/-inf/\"-inf\"/g" probabilities
cat probabilities | python -m json.tool > probabilities.json
