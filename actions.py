from loguru import logger
import sys
import argparse
import os
import shutil
import fnmatch
import json
from subprocess import PIPE, CompletedProcess, run, call, Popen, check_output

parser = argparse.ArgumentParser(description="Modules Release Tool")

parser.add_argument('--cloned-release-repo-dir',
                    action='store',
                    required=True,
                    help="Directory of the cloned release repo.")

parser.add_argument('--cmake-build-dir',
                    action='store',
                    required=True,
                    help="The CMake build directory of the release.")

parser.add_argument('--build-number',
                    action='store',
                    required=True)

parser.add_argument('--repo-url',
                    action='store',
                    required=True,
                    help="The URL of the GitHub repo to create releases in.")

parser.add_argument('--repo-org',
                    action='store',
                    required=True,
                    help="The GitHub organization name of the release repo.")

parser.add_argument('--repo-name',
                    action='store',
                    required=True,
                    help="The GitHub repo name of the release repo.")

parser.add_argument('--dry-run',
                    action='store_true',
                    required=False,
                    help="Dry run, do not upload anything.",
                    default=False)

parser.add_argument('--build-all',
                    action='store_true',
                    required=False,
                    default=False,
                    help="Build all modules.",)

MODULES = {}

def custom_run(args, dry_run):
    if dry_run:
        logger.info("Dry run: %s" % " ".join(args))
        return CompletedProcess(args, 0, "", "")
    return run(args, env=os.environ.copy())


def get_branch():
    return check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()


def get_latest_release_tag():
    '''
    Returns the latest release tag in the repo. 
    Release tags are prefixed with "build-".
    Uses git to get the tags in the repo in current working directory.
    At first, fetches all tags just in case.
    Then returns the latest tag name with prefix "build-".
    Returns None if no tags with prefix "build-" are found.
    '''
    # Fetch all tags
    logger.info("Fetching tags...")
    re = run(["git", "fetch", "--tags"])
    if re.returncode != 0:
        logger.error("Failed to fetch tags.")
        exit(re.returncode)

    # Get the tags using git command
    git_tags = check_output(["git", "tag"]).decode().splitlines()

    # Filter tags with prefix "build-"
    release_tags = [tag for tag in git_tags if tag.startswith(f"build.{get_branch()}")]

    if release_tags:
        # Sort the tags and get the latest one
        sorted_tags = sorted(release_tags, key=lambda tag: int(tag.split("-")[1]))
        latest_tag = sorted_tags[-1]
        return latest_tag
    else:
        return None


def get_list_of_changed_files_between(prev_tag, cur_tag):
    # git diff --name-only prev_tag..cur_tag
    re = run(["git", "diff", "--name-only", f"{prev_tag}..{cur_tag}"], stdout=PIPE, stderr=PIPE, universal_newlines=True)
    if re.returncode != 0:
        logger.error("Failed to get list of changed files.")
        exit(re.returncode)
    return re.stdout.strip().split("\n")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, format="<green>[Modules Release Tool]</green> <level>{time:HH:mm:ss.SSS}</level> <level>{level}</level> <level>{message}</level>")

    args = parser.parse_args()

    with open(f"./modules.json", "r") as f:
        MODULES = json.load(f)
        MODULES_FLAT = dict()
        for type, modules in MODULES.items():
            MODULES_FLAT.update(modules)

    logger.info(f"Target: {args.repo_url}")
    logger.info(f"Build number: {args.build_number}")

    if not os.path.exists(args.cloned_release_repo_dir):
        logger.error(f"Cloned release repo {args.cloned_release_repo_dir} does not exist.")
        exit(1)

    if not os.path.exists(args.cmake_build_dir):
        logger.error(f"CMake build directory {args.cmake_build_dir} does not exist.")
        exit(1)
    
    # Cleanup ./Stage & ./Releases
    if os.path.exists("./Stage"):
        shutil.rmtree("./Stage")
    if os.path.exists("./Releases"):
        shutil.rmtree("./Releases")

    modules_to_release = set()
    if args.build_all:
        modules_to_release.update(MODULES_FLAT.keys())
    else:
        latest_tag = get_latest_release_tag()
        logger.info(f"Latest release tag: {latest_tag}")
        if latest_tag is None:
            logger.info("Including all modules in the release")
            modules_to_release.update(MODULES_FLAT.keys())
        else:
            changed_files = get_list_of_changed_files_between(latest_tag, "HEAD")
            logger.debug(f"Changed files: {changed_files}")
            for module_name, module_info in MODULES_FLAT.items():
                for dep in module_info["deps"]:
                    for changed_file in changed_files:
                        if fnmatch.fnmatch(changed_file, dep):
                            modules_to_release.add(module_name)
                            break

    if len(modules_to_release) == 0:
        logger.info("None of the modules have changed. No need to release.")
        exit(0)

    logger.info(f"Modules to release: {modules_to_release}")

    # Run python release.py --gh-release-repo="{args.gh_release_repo}" make --build-number="{args.build_number}"  
    #       --release-target={target_name} --cmake-build-dir={args.cmake_build_dir} --module-dir="{module_folder}" 
    # TODO: release.py is not parallelizable yet, so we run it sequentially for each module. Make it parallel.
    ok = True
    for module_name in modules_to_release:
        module_info = MODULES_FLAT[module_name]
        proc_args = ["python", "release.py",
                      "make", "--build-number", args.build_number, 
                      "--release-target", module_info["target_name"], 
                      "--cmake-build-dir", args.cmake_build_dir, 
                      "--module-dir", f"{'Plugins' if module_name in MODULES['plugins'] else 'Subsystems'}/{module_info['path']}"]
        logger.info(f"Creating module release for {module_name} with command: {' '.join(proc_args)}")
        re = custom_run(proc_args, args.dry_run)
        if re.returncode != 0:
            logger.error(f"Failed to release module {module_name}")
            ok = False

    if not ok:
        exit(1)
    
    # Upload releases
    logger.info(f"Uploading releases of modules {modules_to_release} to {args.repo_url}")
    re = custom_run(["python", "release.py", "upload", "--cloned-release-repo", args.cloned_release_repo_dir, 
              "--repo-url", args.repo_url, "--repo-org", args.repo_org, "--repo-name", args.repo_name],
             args.dry_run)
    if re.returncode != 0:
        logger.error(f"Failed to upload releases")
        exit(re.returncode)

    # Create tag to specify the latest release
    tag_msg = f"Build {args.build_number}\n\n"
    tag_msg += "Modules in this release:\n"
    for module_name in modules_to_release:
        tag_msg += f"\n{module_name}"
    re = custom_run(["git" ,"tag", "-a", f"build.{get_branch()}-{args.build_number}", "-m", tag_msg], args.dry_run)
    if re.returncode != 0:
        logger.error(f"Failed to create tag")
        exit(re.returncode)
    re = custom_run(["git", "push", "--tags"], args.dry_run)
    if re.returncode != 0:
        logger.error(f"Failed to push tags")
        exit(re.returncode)

    logger.info("Done.")
