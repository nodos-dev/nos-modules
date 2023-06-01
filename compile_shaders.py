#!/usr/bin/env python3

from subprocess import PIPE, run, call, Popen
import platform
from sys import stdout, stderr
import os
import sys
from logger import logger

import threading

path = str(os.path.dirname(__file__))

def embed_binary(filepath):
    result = run([f"{path}/tools/{platform.system()}/bin2header.exe", filepath], stdout=PIPE, stderr=PIPE, universal_newlines=True)
    if result.returncode != 0:
        logger.warning(f"Failed to embed {filepath}")
        exit(result.returncode)
    else:
        os.replace(filepath + ".h", filepath + ".dat")

def compile_to_spv(filepath):
    logger.info(f"Compiling {filepath}")
    re = run([f"{path}/tools/{platform.system()}/glslc", "-O", "-g", "-o",  f"{filepath}_.spv", filepath], stdout=stdout, stderr=stderr, universal_newlines=True)
    if re.returncode != 0:
        logger.error(f"Failed to compile {filepath}")
        return re.returncode
    else:
        re = run(["spirv-opt", "-O", "-o",  f"{filepath}.spv", f"{filepath}_.spv"], stdout=stdout, stderr=stderr, universal_newlines=True)
        os.remove(f"{filepath}_.spv")
        if re.returncode != 0:
            logger.error(f"Failed to optimize {filepath}")
            return re.returncode
        else:
            embed_binary(f"{filepath}.spv")
            os.remove(f"{filepath}.spv")
    return 0

def compile_shaders():
    compile_threads = []
    for root, dirs, files in os.walk(path):
        for filepath in files:
            if filepath.endswith(".frag") or filepath.endswith(".vert") or filepath.endswith(".comp"):
                fullpath = os.path.join(root, filepath)
                compiled = fullpath + ".spv.dat"
                if os.path.exists(compiled) and os.stat(fullpath).st_mtime <= os.stat(compiled).st_mtime:
                    continue
                th = threading.Thread(target=compile_to_spv,args=(fullpath,))
                th.start()
                compile_threads.append(th)
    for th in compile_threads:
        th.join()


compile_shaders()
