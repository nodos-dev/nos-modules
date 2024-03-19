# Copyright MediaZ Teknoloji A.S. All Rights Reserved.

from loguru import logger
import sys
import argparse
import os
import shutil
import fnmatch
import json
from subprocess import PIPE, CompletedProcess, run, call, Popen

parser = argparse.ArgumentParser(description="Generates releases for Nodos modules")

subparsers = parser.add_subparsers(dest="command", help="Subcommands")

make = subparsers.add_parser("make", help="Create a release zip for a module")

make.add_argument('--build-number',
                    action='store',
                    required=True,
                    help="The version string of the release.")

make.add_argument('--module-name',
                    action='store',
                    required=True,
                    help="Which module should I create release zip of?")

make.add_argument('--cmake-build-dir',
                    action='store',
                    required=True,
                    help="The CMake build directory of the release.")

make.add_argument('--module-dir',
                    action='store',
                    required=True,
                    help="Module directory that contains binaries, and config files. Files in this folder will be used to create the release zip.")

upload = subparsers.add_parser("upload", help="Create a release at GitHub")

upload.add_argument('--cloned-release-repo-dir', 
                    action='store',
                    required=True,
                    help="Directory of the cloned release repo.")

upload.add_argument('--repo-url',
                    action='store',
                    required=True,
                    help="The URL of the GitHub repo to create releases in.")

upload.add_argument('--repo-org',
                    action='store',
                    required=True,
                    help="The GitHub organization name of the release repo.")

upload.add_argument('--repo-name',
                    action='store',
                    required=True,
                    help="The GitHub repo name of the release repo.")

upload.add_argument('--dry-run',
                    action='store_true',
                    required=False,
                    help="Dry run. Do not upload anything to GitHub. Only print the commands that would be executed.",
                    default=False)

def custom_run(args, dry_run):
    if dry_run:
        logger.info("Dry run: %s" % " ".join(args))
        return CompletedProcess(args, 0, "", "")
    return run(args, env=os.environ.copy())


def get_api_version(api_header, api_name):
    sdk_dir = os.getenv("NODOS_SDK_DIR")
    if sdk_dir is None or sdk_dir == "":
        logger.error("NODOS_SDK_DIR is not set.")
        exit(1)
    nos_api_h = os.path.join(sdk_dir, "include", "Nodos", api_header)
    if not os.path.exists(nos_api_h):
        logger.error("NODOS_SDK_DIR is not set correctly.")
        exit(1)
    with open(nos_api_h, "r") as f:
        major = None
        minor = None
        patch = None
        for line in f.readlines():
            # 👏👏👏
            if line.startswith(f"#define NOS_{api_name}_API_VERSION_MAJOR"):
                major = int(line.split(" ")[-1])
            elif line.startswith(f"#define NOS_{api_name}_API_VERSION_MINOR"):
                minor = int(line.split(" ")[-1])
            elif line.startswith(f"#define NOS_{api_name}_API_VERSION_PATCH"):
                patch = int(line.split(" ")[-1])
        if major is None or minor is None or patch is None:
            logger.error(f"Failed to parse NOS_{api_name}_API_VERSION")
            exit(1)
        return {"major": major, "minor": minor, "patch": patch}


def get_module_info(target_name):
    for type, modules in MODULES.items():
        if type == "files":
            continue
        for _, module in modules.items():
            if module["target_name"] == target_name:
                return module
    return None

def get_module(module_name):
    for type, modules in MODULES.items():
        if type == "files":
            continue
        for k, module in modules.items():
            if k == module_name:
                return module
    return None

def make_release(args):
    logger.debug(f"Creating release zip for {args.module_name}")
    logger.info(f"Target: {args.module_name}")
    
    logger.info(f"Building {args.module_name}")
    module_info = get_module(args.module_name)

    if "target_name" in module_info.keys():
        release_target = module_info["target_name"]
        re = run(["cmake", "--build", args.cmake_build_dir, "--config", "Release", "--target", release_target], universal_newlines=True)
        if re.returncode != 0:
            logger.error(f"Failed to build {release_target}")
            exit(re.returncode)
        logger.info(f"Built {release_target} successfully")

    logger.debug(f"Creating a release zip for {args.module_name}")

    files_to_include = MODULES["files"]
    
    if module_info is not None and "files" in module_info.keys():
        files_to_include.extend(module_info["files"])
    logger.info(f"Collecting files: {files_to_include}")

    collected_files = []
    nos_module_file = None
    for root, dirs, files in os.walk(args.module_dir):
        for file in files:
            full_path = os.path.join(root, file)
            # Remove module_dir from path
            rel_path = os.path.relpath(full_path, args.module_dir)
            if any([fnmatch.fnmatch(rel_path, pattern) for pattern in files_to_include]):
                collected_files.append(full_path)
            # Find .noscfg file:
            if file.endswith(".noscfg") or file.endswith(".nossys"):
                nos_module_file = full_path

    if nos_module_file is None:
        logger.error(f"Failed to find .noscfg or .nossys file in {args.module_dir}")
        exit(1)

    nos_module_json = None
    with open(nos_module_file, "r") as f:
        nos_module_json = json.load(f)
    
    module_version = nos_module_json["info"]["id"]["version"]
    module_version = f"{module_version}.b{args.build_number}"
    nos_module_json["info"]["id"]["version"] = module_version
    with open(nos_module_file, "w") as f:
        json.dump(nos_module_json, f, indent=4)

    os.makedirs("Stage", exist_ok=True)
    logger.debug(f"Collected files: {collected_files}")
    logger.info(f"Copying files to staging folder")
    # Copy files to "./Stage", while preserving the directory structure. Eg. "./Stage/..."
    # Create directories if they don't exist.
    for file in collected_files:
        target_dir = os.path.join("Stage", os.path.dirname(os.path.relpath(file, args.module_dir)))
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(file, target_dir)
    
    logger.info(f"Creating release zip")
    module_name = nos_module_json["info"]["id"]["name"]
    zip_name = f"{module_name}-{module_version}"
    shutil.make_archive(zip_name, "zip", "Stage")
    os.makedirs("Releases", exist_ok=True)
    shutil.move(f"{zip_name}.zip", os.path.join("Releases", f"{zip_name}.zip"))
    logger.info(f"Created release zip: {os.path.join('Releases', f'{zip_name}.zip')}")

    logger.info(f"Cleaning up")
    shutil.rmtree("Stage")


def upload_releases(repo_url, org_name, repo_name, cloned_release_repo, dry_run):
    repo_org_name = f"{org_name}/{repo_name}"
    # Collect zip files under "./Releases"
    zip_files = []
    for root, dirs, files in os.walk("Releases"):
        for file in files:
            if file.endswith(".zip"):
                zip_files.append(os.path.join(root, file))
    logger.debug(f"Found zip files: {zip_files}")
    logger.info(f"GitHub Release: Pushing release artifacts to repo {repo_org_name}")
    artifacts = zip_files
    # https://github.com/mediaz/nodos/releases/download/v0.1.0.b1769/Nodos-SDK-v0.1.0.b1769.zip

    for artifact in artifacts:
        os.chdir(cloned_release_repo)

        filename = os.path.basename(artifact)
        module_name = filename.split("-")[0]
        module_version = filename.split("-")[1].split(".zip")[0]
        tag = f"{module_name}-{module_version}"

        logger.info(f"Updating index file for {module_name} {module_version}")
        os.makedirs(f"{module_name}", exist_ok=True)
        index = { "name": module_name, "releases": [] }
        if os.path.exists(f"{module_name}/index.json"):
            with open(f"{module_name}/index.json", "r") as f:
                index = json.load(f)
        release_zip_download_url = f"{repo_url}/releases/download/{tag}/{filename}"
        release_info = { "version": module_version, "url": release_zip_download_url }
        if module_name in MODULES["plugins"]:
            release_info["plugin_api_version"] = get_api_version("PluginAPI.h", "PLUGIN")
        elif module_name in MODULES["subsystems"]:
            release_info["subsystem_api_version"] = get_api_version("SubsystemAPI.h", "SUBSYSTEM")

        index["releases"].insert(0, release_info)
        with open(f"{module_name}/index.json", "w") as f:
            json.dump(index, f, indent=4)

        author_email = os.getenv("GIT_EMAIL")
        author_name = os.getenv("GH_USERNAME")
        if author_email is None:
            logger.error("GIT_EMAIL not set")
            exit(1)
        if author_name is None:
            logger.error("GH_USERNAME not set")
            exit(1)

        re = custom_run(["git", "config", "user.email", author_email], dry_run)
        if re.returncode != 0:
            logger.error(f"Failed to set git user email: {re.stderr}")
            exit(re.returncode)
        re = custom_run(["git", "config", "user.name", author_name], dry_run)
        if re.returncode != 0:
            logger.error(f"Failed to set git user name: {re.stderr}")
            exit(re.returncode)

        # Commit the result and create a release
        re = custom_run(["git", "add", f"{module_name}/index.json"], dry_run)
        if re.returncode != 0:
            logger.error(f"Failed to add index file: {re.stderr}")
            exit(re.returncode)
        re = custom_run(["git", "commit", "-m", f"Update index file for {module_name} {module_version}"], dry_run)
        if re.returncode != 0:
            logger.error(f"Failed to commit index file: {re.stderr}")
            exit(re.returncode)
        re = custom_run(["git", "push"], dry_run)
        if re.returncode != 0:
            logger.error(f"Failed to push: {re.stderr}")
            exit(re.returncode)

        os.chdir("..")

        re = custom_run(["gh", "release", "create", tag, artifact, "--repo", repo_org_name, "--title", tag], dry_run)
        if re.returncode != 0:
            logger.error(f"Failed to create release: {re.stderr}")
            exit(re.returncode)
        logger.info(f"Created release: {tag}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, format="<green>[Modules Release Tool]</green> <level>{time:HH:mm:ss.SSS}</level> <level>{level}</level> <level>{message}</level>")
    args = parser.parse_args()

    global MODULES
    with open(f"./modules.json", "r") as f:
        MODULES = json.load(f)

    if args.command == "make":
        make_release(args)
    elif args.command == "upload":
        upload_releases(args.repo_url,
                        args.repo_org,
                        args.repo_name,
                        args.cloned_release_repo_dir,
                        args.dry_run)