''' Collect op registry information. '''

from __future__ import print_function
import sys
import logging
from ast import RegisterLiteOpParser

if len(sys.argv) != 6:
    print("Error: parse_op_registry.py requires four inputs!")
    exit(1)
ops_list_path = sys.argv[1]
dest_path = sys.argv[2]
minops_list_path = sys.argv[3]
tailored = sys.argv[4]
with_extra = sys.argv[5]
out_lines = [
    '#pragma once',
    '#include "paddle_lite_factory_helper.h"',
    '',
]

paths = set()
for line in open(ops_list_path):
    paths.add(line.strip())

if tailored == "ON":
    minlines = set()
    with open(minops_list_path) as fd:
        for line in fd:
            minlines.add(line.strip())
for path in paths:
    str_info = open(path.strip()).read()
    op_parser = RegisterLiteOpParser(str_info)
    ops = op_parser.parse(with_extra)
    for op in ops:
        if tailored == "ON":
            if op not in minlines: continue
        out = "USE_LITE_OP(%s);" % op
        out_lines.append(out)

with open(dest_path, 'w') as f:
    logging.info("write op list to %s" % dest_path)
    f.write('\n'.join(out_lines))
