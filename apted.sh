#! /bin/bash
cat $1 | while read line; do echo $line|tr '\n' '\t'; python -m apted -t $line; done