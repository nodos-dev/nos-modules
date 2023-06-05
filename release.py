from loguru import logger
import sys
import argparse
import os
import shutil
import fnmatch
import json
from subprocess import PIPE, run, call, Popen

parser = argparse.ArgumentParser(description="Generate releases for MZ Plugin Bundle")

parser.add_argument('--gh-release-repo',
                    action='store',
                    required=True,
                    help="The repo of the release.")

subparsers = parser.add_subparsers(dest="command", help="Subcommands")

make = subparsers.add_parser("make", help="Create a release zip for a plugin")

make.add_argument('--build-number',
                    action='store',
                    required=True,
                    help="The version string of the release.")

make.add_argument('--release-target',
                    action='store',
                    required=True,
                    help="Which plugin should I create release zip of?")

make.add_argument('--cmake-build-dir',
                    action='store',
                    required=True,
                    help="The CMake build directory of the release.")

make.add_argument('--plugin-dir',
                    action='store',
                    required=True,
                    help="Plugin directory that contains binaries, and config files. Files in this folder will be used to create the release zip.")

make.add_argument('--exclude',
                    action='store',
                    required=False,
                    default="*CMakeLists.txt,*.cpp,*.cc,*.c,*.h,*.hxx,*.hpp,*.frag,*.glsl,*.vert,*.comp,*.spv*,*.dat,*.pdb",
                    help="Comma separated filenames and wildcards to exclude files from the release zip.")

upload = subparsers.add_parser("upload", help="Create a release at GitHub")

upload.add_argument('--cloned-release-repo', 
                    action='store',
                    required=True,
                    help="Directory of the cloned release repo.")

def make_release(args):
    logger.debug(f"Creating release for {args.release_target} in {args.gh_release_repo}")
    logger.info(f"Target: {args.release_target}")
    logger.info(f"Repo: {args.gh_release_repo}")

    logger.info(f"Building {args.release_target}")
    re = run(["cmake", "--build", args.cmake_build_dir, "--config", "Release", "--target", args.release_target], universal_newlines=True)
    if re.returncode != 0:
        logger.error(f"Failed to build {args.release_target}")
        exit(re.returncode)
    logger.info(f"Built {args.release_target} successfully")

    logger.debug(f"Creating a release zip for {args.release_target}")
    
    collected_files = []
    mzcfg_file = None
    for root, dirs, files in os.walk(args.plugin_dir):
        for file in files:
            full_path = os.path.join(root, file)
            if not any([fnmatch.fnmatch(full_path, pattern) for pattern in args.exclude.split(",")]):
                collected_files.append(full_path)
            # Find .mzcfg file:
            if file.endswith(".mzcfg"):
                mzcfg_file = full_path

    if mzcfg_file is None:
        logger.error(f"Failed to find .mzcfg file in {args.plugin_dir}")
        exit(1)

    mzcfg = None
    with open(mzcfg_file, "r") as f:
        mzcfg = json.load(f)
    
    plugin_version = mzcfg["info"]["id"]["version"]
    plugin_version = f"{plugin_version}.b{args.build_number}"
    mzcfg["info"]["id"]["version"] = plugin_version
    with open(mzcfg_file, "w") as f:
        json.dump(mzcfg, f, indent=4)

    os.makedirs("Stage", exist_ok=True)
    logger.debug(f"Collected files: {collected_files}")
    logger.info(f"Copying files to staging folder")
    # Copy files to "./Stage", while preserving the directory structure. Eg. "./Stage/PluginName/..."
    # Create directories if they don't exist.
    for file in collected_files:
        os.makedirs(os.path.join("Stage", os.path.dirname(file)), exist_ok=True)
        dst = os.path.join("Stage", file)
        logger.debug(f"Copying {file} to {dst}")
        shutil.copy(file, dst)

    logger.info(f"Creating release zip")
    plugin_name = mzcfg["info"]["id"]["name"]
    zip_name = f"{plugin_name}-{plugin_version}"
    shutil.make_archive(zip_name, "zip", "Stage")
    os.makedirs("Releases", exist_ok=True)
    shutil.move(f"{zip_name}.zip", os.path.join("Releases", f"{zip_name}.zip"))
    logger.info(f"Created release zip: {os.path.join('Releases', f'{zip_name}.zip')}")


def upload_releases(gh_release_repo, cloned_release_repo):
    # Collect zip files under "./Releases"
    zip_files = []
    for root, dirs, files in os.walk("Releases"):
        for file in files:
            if file.endswith(".zip"):
                zip_files.append(os.path.join(root, file))
    logger.debug(f"Found zip files: {zip_files}")
    logger.info(f"GitHub Release: Pushing release artifacts to repo {gh_release_repo}")
    artifacts = zip_files
    # https://github.com/mediaz/mediaz/releases/download/v0.1.0.b1769/mediaZ-SDK-v0.1.0.b1769.zip
    for artifact in artifacts:
        os.chdir(cloned_release_repo)

        filename = os.path.basename(artifact)
        plugin_name = filename.split("-")[0]
        plugin_version = filename.split("-")[1].split(".zip")[0]
        tag = f"{plugin_name}-{plugin_version}"
        logger.info(f"Updating index file for {plugin_name} {plugin_version}")
        os.makedirs(f"{plugin_name}", exist_ok=True)
        index = { "name": plugin_name, "releases": [] }
        if os.path.exists(f"{plugin_name}/index.json"):
            with open(f"{plugin_name}/index.json", "r") as f:
                index = json.load(f)
        release_zip_download_url = f"{gh_release_repo}/releases/download/{tag}/{filename}"
        index["releases"].insert(0, { "version": plugin_version, "url": release_zip_download_url })
        with open(f"{plugin_name}/index.json", "w") as f:
            json.dump(index, f, indent=4)

        author_email = os.getenv("GIT_EMAIL")
        author_name = os.getenv("GH_USERNAME")
        if author_email is None:
            logger.error("GIT_EMAIL not set")
            exit(1)
        if author_name is None:
            logger.error("GH_USERNAME not set")
            exit(1)

        re = run(["git", "config", "user.email", author_email], capture_output=True, text=True, env=os.environ.copy())
        if re.returncode != 0:
            logger.error(f"Failed to set git user email: {re.stderr}")
            exit(re.returncode)
        re = run(["git", "config", "user.name", author_name], capture_output=True, text=True, env=os.environ.copy())
        if re.returncode != 0:
            logger.error(f"Failed to set git user name: {re.stderr}")
            exit(re.returncode)

        # Commit the result and create a release
        re = run(["git", "add", f"{plugin_name}/index.json"], capture_output=True, text=True, env=os.environ.copy())
        if re.returncode != 0:
            logger.error(f"Failed to add index file: {re.stderr}")
            exit(re.returncode)
        re = run(["git", "commit", "-m", f"Update index file for {plugin_name} {plugin_version}"], capture_output=True, text=True, env=os.environ.copy())
        if re.returncode != 0:
            logger.error(f"Failed to commit index file: {re.stderr}")
            exit(re.returncode)
        re = run(["git", "push"], capture_output=True, text=True, env=os.environ.copy())

        os.chdir("..")

        re = run(["gh", "release", "create", tag, *artifacts, "--repo", args.gh_release_repo, "--title", tag], 
                    capture_output=True, text=True, env=os.environ.copy())
        if re.returncode != 0:
            logger.error(f"Failed to create release: {re.stderr}")
            exit(re.returncode)
        logger.info(f"Created release: {tag}")

    if re.returncode != 0:
        logger.error(f"Failed to create release: {re.stderr}")
        exit(re.returncode)
    logger.info(f"Created release: {tag}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, format="<green>[MZ Plugin Bundle Source]</green> <level>{time:HH:mm:ss.SSS}</level> <level>{level}</level> <level>{message}</level>")
    args = parser.parse_args()
    if args.command == "make":
        make_release(args)
    elif args.command == "upload":
        upload_releases(args.gh_release_repo, args.cloned_release_repo)