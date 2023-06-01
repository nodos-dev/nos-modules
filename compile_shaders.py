#!/usr/bin/env python3

from subprocess import PIPE, run, call, Popen
import platform
from sys import stdout, stderr
import os
import sys
import threading

#!/usr/bin/env python3

import logging
from subprocess import PIPE, run, call, Popen
import platform
from sys import stdout, stderr
import functools
print = functools.partial(print, flush=True)

class CustomFormatter(logging.Formatter):
    white = "\x1b[37m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    green = "\x1b[32;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s [%(name)s] %(levelname)s\t%(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        None: format,
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def duration_human_readable(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.2f} minutes"
    hours = minutes / 60
    return f"{hours:.2f} hours"


logger = logging.getLogger("MediaZ Source")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

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
                    logger.info(f"{filepath} is up to date.")
                    continue
                th = threading.Thread(target=compile_to_spv,args=(fullpath,))
                th.start()
                compile_threads.append(th)
    for th in compile_threads:
        th.join()


compile_shaders()
