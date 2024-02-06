#!/usr/bin/env python3

from subprocess import PIPE, CompletedProcess, run, call, Popen
from sys import stdout, stderr
import os
import argparse
from loguru import logger
import re;
from pathlib import Path
import re

ap = argparse.ArgumentParser()
ap.add_argument("-folder", required=True, help="Folder to search cuda kernels")

def generate_cuda_kernel_as_char_array(input_folder):

    print("Processing directory:", Path(input_folder)) 
    #for ptx_file_path in Path(input_folder).rglob('*.cu'): Add .cu compilation too "nvcc -keep *.cu"

    for ptx_file_path in Path(input_folder).rglob('*.ptx'):
        print("PTX File found:", ptx_file_path)
        output_dir_embedded = str(ptx_file_path) + ".h"
        with open(ptx_file_path, 'r') as file:
            ptx_str = file.read()
        function_name = re.findall("entry\s+([A-Za-z_][A-Za-z0-9_]*)", ptx_str)[0]
        embedded_str = "#ifndef "+function_name.upper() + "_INCLUDED\n"
        embedded_str += "#define "+function_name.upper() + "_INCLUDED\n"
        embedded_str += f'static const char {function_name}[] = R"({ptx_str})";\n'
        embedded_str += "#endif //"+function_name.upper() + "_INCLUDED"

        with open(output_dir_embedded, 'w') as generated_file:
            generated_file.write(embedded_str)
        print("PTX header generated")
def main():
    args = vars(ap.parse_args())
    print(args.get("folder").strip('"'))
    generate_cuda_kernel_as_char_array(args.get("folder").strip('"'))
    print("...Generating completed...")

if __name__ == '__main__':
    main()
