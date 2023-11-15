from loguru import logger
import sys
import argparse
import os
import shutil
import fnmatch
import json
from subprocess import PIPE, CompletedProcess, run, call, Popen

parser = argparse.ArgumentParser(description="Generate releases for MZ Plugin Bundle")

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
                    default="*CMakeLists.txt,*.cpp,*.cc,*.c,*.h,*.hxx,*.hpp,*.dat,*.pdb",
                    help="Comma separated filenames and wildcards to exclude files from the release zip.")

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


def get_plugin_api_version():
    sdk_dir = os.getenv("MZ_SDK_DIR")
    if sdk_dir is None or sdk_dir == "":
        logger.error("MZ_SDK_DIR is not set.")
        exit(1)
    mz_plugin_api_h = os.path.join(sdk_dir, "include", "MediaZ", "PluginAPI.h")
    if not os.path.exists(mz_plugin_api_h):
        logger.error("MZ_SDK_DIR is not set correctly.")
        exit(1)
    with open(mz_plugin_api_h, "r") as f:
        major = None
        minor = None
        patch = None
        for line in f.readlines():
            # 👏👏👏
            if line.startswith("#define MZ_PLUGIN_API_VERSION_MAJOR"):
                major = int(line.split(" ")[-1])
            elif line.startswith("#define MZ_PLUGIN_API_VERSION_MINOR"):
                minor = int(line.split(" ")[-1])
            elif line.startswith("#define MZ_PLUGIN_API_VERSION_PATCH"):
                patch = int(line.split(" ")[-1])
        if major is None or minor is None or patch is None:
            logger.error("Failed to parse MZ_PLUGIN_API_VERSION")
            exit(1)
        return {"major": major, "minor": minor, "patch": patch}


def make_release(args):
    logger.debug(f"Creating release zip for {args.release_target}")
    logger.info(f"Target: {args.release_target}")

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
    # Copy files to "./Stage", while preserving the directory structure. Eg. "./Stage/..."
    # Create directories if they don't exist.
    for file in collected_files:
        target_dir = os.path.join("Stage", os.path.dirname(os.path.relpath(file, args.plugin_dir)))
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(file, target_dir)
    
    logger.info(f"Creating release zip")
    plugin_name = mzcfg["info"]["id"]["name"]
    zip_name = f"{plugin_name}-{plugin_version}"
    shutil.make_archive(zip_name, "zip", "Stage")
    os.makedirs("Releases", exist_ok=True)
    shutil.move(f"{zip_name}.zip", os.path.join("Releases", f"{zip_name}.zip"))
    logger.info(f"Created release zip: {os.path.join('Releases', f'{zip_name}.zip')}")

    logger.info(f"Cleaning up")
    shutil.rmtree("Stage")


def upload_releases(repo_url, org_name, repo_name, cloned_release_repo, dry_run):
    plugin_api_ver = get_plugin_api_version()
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
        release_zip_download_url = f"{repo_url}/releases/download/{tag}/{filename}"
        index["releases"].insert(0, { "version": plugin_version, "url": release_zip_download_url, "plugin_api_version": plugin_api_ver })
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

        re = custom_run(["git", "config", "user.email", author_email], dry_run)
        if re.returncode != 0:
            logger.error(f"Failed to set git user email: {re.stderr}")
            exit(re.returncode)
        re = custom_run(["git", "config", "user.name", author_name], dry_run)
        if re.returncode != 0:
            logger.error(f"Failed to set git user name: {re.stderr}")
            exit(re.returncode)

        # Commit the result and create a release
        re = custom_run(["git", "add", f"{plugin_name}/index.json"], dry_run)
        if re.returncode != 0:
            logger.error(f"Failed to add index file: {re.stderr}")
            exit(re.returncode)
        re = custom_run(["git", "commit", "-m", f"Update index file for {plugin_name} {plugin_version}"], dry_run)
        if re.returncode != 0:
            logger.error(f"Failed to commit index file: {re.stderr}")
            exit(re.returncode)
        re = custom_run(["git", "push"], dry_run)

        os.chdir("..")

        re = custom_run(["gh", "release", "create", tag, artifact, "--repo", repo_org_name, "--title", tag], dry_run)
        if re.returncode != 0:
            logger.error(f"Failed to create release: {re.stderr}")
            exit(re.returncode)
        logger.info(f"Created release: {tag}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, format="<green>[Plugin Bundle Release Tool]</green> <level>{time:HH:mm:ss.SSS}</level> <level>{level}</level> <level>{message}</level>")
    args = parser.parse_args()
    if args.command == "make":
        make_release(args)
    elif args.command == "upload":
        upload_releases(args.repo_url,
                        args.repo_org,
                        args.repo_name,
                        args.cloned_release_repo_dir,
                        args.dry_run)