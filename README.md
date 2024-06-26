# Modules

This folder contains the Nodos modules that are distributed with Nodos.

## Build Instructions
1. Download latest Nodos release from [nodos.dev](https://nodos.dev)
2. Clone the repository under Nodos workspace Module directory
```bash
git clone https://github.com/mediaz/nos-modules.git --recurse-submodules Module/nos-modules
```
3. Generate project files from workspace root directory using CMake:
```bash
cmake -S ./Toolchain/CMake -B Build
```
4. Build the project:
```bash
cmake --build Build
```

## Structure
A plugin structure is as follows:

```
SomePlugin/
├─ SomePlugin.noscfg (or .nossys if a subsystem)
├─ Binaries/ (shipped)
│  ├─ SomePlugin.dll
├─ Config/ (shipped)
│  ├─ SomePlugin.fbs
│  ├─ SomePlugin.nosdef
├─ Source/ (example)
│  ├─ SomePlugin.cpp
```
