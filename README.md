# Plugins

This folder contains the plugins that will be distributed with Nodos.

## Build Instructions
```bash
pip install -r requirements.txt
cmake -S . -B Build
cmake --build Build
```

## Structure
A plugin structure is as follows:

```
SomePlugin/
├─ Binaries/ (shipped)
│  ├─ SomePlugin.dll
├─ Config/ (shipped)
│  ├─ SomePlugin.fbs
│  ├─ SomePlugin.noscfg
│  ├─ SomePlugin.nosdef
├─ Source/ (example)
│  ├─ SomePlugin.cpp
```

## Current Plugin SDK Status

Nodos Plugin SDK is active in development. It's interface is not stable and contains references to C++ standard library. 
