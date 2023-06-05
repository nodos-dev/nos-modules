from loguru import logger
import sys
import argparse
import os
import shutil
import fnmatch
import json
from subprocess import PIPE, run, call, Popen

parser = argparse.ArgumentParser(description="MZ Plugin Bundle Release Tool")

parser.add_argument('--gh-release-repo',
                    action='store',
                    required=True,
                    help="URL to the release repo that contains index files and release artifacts.")

parser.add_argument('--cloned-release-repo',
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

PLUGINS = {
   "AJA":{
      "target_name":"mzAJA",
      "path": "mzAJA",
      "deps":[
         "mzAJA/*"
      ]
   },
   "Cyclorama":{
      "target_name":"mzCyclorama",
      "path": "mzCyclorama",
      "deps":[
         "mzCyclorama/*"
      ]
   },
   "mz.filters":{
      "target_name":"mzFilters",
      "path": "mzFilters",
      "deps":[
         "mzFilters/*",
         "Shaders/*"
      ]
   },
   "mz.math":{
      "target_name":"mzMath",
      "path": "mzMath",
      "deps":[
         "mzMath/*"
      ]
   },
   "mz.realitykeyer":{
      "target_name":"mzRealityKeyer",
      "path": "mzRealityKeyer",
      "deps":[
         "mzRealityKeyer/*",
         "Shaders/ShaderCommon.glsl"
      ]
   },
   "mz.track":{
      "target_name":"mzTrack",
      "path": "mzTrack",
      "deps":[
         "mzTrack/*"
      ]
   },
   "mz.utilities":{
      "target_name":"mzUtilities",
      "path": "mzUtilities",
      "deps":[
         "mzUtilities/*",
         "Shaders/*"
      ]
   },
   "zd.frustum":{
      "target_name":"zdFrustum",
      "path": "zdFrustum",
      "deps":[
         "zdFrustum/*"
      ]
   }
}

def get_latest_release_tag():
    # git fetch --prune --tags --recurse-submodules=no https://${{ env.GH_USERNAME }}:${{ secrets.CI_TOKEN }}@github.com/mediaz/mediaz.git
    # git describe --match "build-*" --abbrev=0 --tags $(git rev-list --tags --max-count=1)
    re = run(["git", "fetch", "--prune", "--tags", "--recurse-submodules=no"], env=os.environ.copy())
    if re.returncode != 0:
        logger.error("Failed to fetch tags!")
        exit(re.returncode)
    re = run(["git", "describe", "--match", "build-*", "--abbrev=0", "--tags", "$(git rev-list --tags --max-count=1)"], env=os.environ.copy())
    if re.returncode != 0:
        logger.warning("No release tag found.")
        return None
    return re.stdout.strip()


def get_list_of_changed_files_between(prev_tag, cur_tag):
    # git diff --name-only prev_tag..cur_tag
    re = run(["git", "diff", "--name-only", f"{prev_tag}..{cur_tag}"], stdout=PIPE, stderr=PIPE, universal_newlines=True)
    if re.returncode != 0:
        logger.error("Failed to get list of changed files.")
        exit(re.returncode)
    return re.stdout.strip().split("\n")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, format="<green>[Plugin Bundle Release Tool]</green> <level>{time:HH:mm:ss.SSS}</level> <level>{level}</level> <level>{message}</level>")

    args = parser.parse_args()
    logger.info(f"Target: {args.gh_release_repo}")
    logger.info(f"Build number: {args.build_number}")

    if not os.path.exists(args.cloned_release_repo):
        logger.error(f"Cloned release repo {args.cloned_release_repo} does not exist.")
        exit(1)

    if not os.path.exists(args.cmake_build_dir):
        logger.error(f"CMake build directory {args.cmake_build_dir} does not exist.")
        exit(1)
    
    # Cleanup ./Stage & ./Releases
    if os.path.exists("./Stage"):
        shutil.rmtree("./Stage")
    if os.path.exists("./Releases"):
        shutil.rmtree("./Releases")

    latest_tag = get_latest_release_tag()
    logger.info(f"Latest release tag: {latest_tag}")
    plugins_to_release = []
    if latest_tag is None:
        logger.info("Including all plugins in the release")
        plugins_to_release = list(PLUGINS.keys())
    else:
        changed_files = get_list_of_changed_files_between(latest_tag, "HEAD")
        logger.debug(f"Changed files: {changed_files}")
        for plugin_name, plugin_info in PLUGINS.items():
            for dep in plugin_info["deps"]:
                for changed_file in changed_files:
                    if fnmatch.fnmatch(changed_file, dep):
                        plugins_to_release.append(plugin_name)
                        break

    logger.info(f"Plugins to release: {plugins_to_release}")

    # Run python release.py --gh-release-repo="{args.gh_release_repo}" make --build-number="{args.build_number}"  
    #       --release-target={target_name} --cmake-build-dir={args.cmake_build_dir} --plugin-dir="{plugin_folder}" 
    # at parallel & print their output if failed. Max 4 processes at a time.
    ok = True
    for plugin_name in plugins_to_release:
        plugin_info = PLUGINS[plugin_name]
        proc_args = ["python", "release.py", 
                      "--gh-release-repo", args.gh_release_repo, 
                      "make", "--build-number", args.build_number, 
                      "--release-target", plugin_info["target_name"], 
                      "--cmake-build-dir", args.cmake_build_dir, 
                      "--plugin-dir", plugin_info["path"]]
        logger.info(f"Creating plugin release for {plugin_name} with command: {' '.join(proc_args)}")
        re = run(proc_args, env=os.environ.copy())
        if re.returncode != 0:
            logger.error(f"Failed to release plugin {plugin_name}")
            ok = False

    if not ok:
        exit(1)
    
    # Upload releases
    logger.info(f"Uploading releases of plugins {plugins_to_release} to {args.gh_release_repo}")
    re = run(["python", "release.py", "--gh-release-repo", args.gh_release_repo, "upload", "--cloned-release-repo", args.cloned_release_repo],
             env=os.environ.copy())
    if re.returncode != 0:
        logger.error(f"Failed to upload releases")
        exit(re.returncode)

    # Create tag to specify the latest release
    tag_msg = f"Build {args.build_number}\n\n"
    tag_msg += "Plugins in this release:\n"
    for plugin_name in plugins_to_release:
        tag_msg += f"\n{plugin_name}"
    re = run(["git" ,"tag", "-a", f"build-{args.build_number}", "-m", tag_msg], env=os.environ.copy())
    if re.returncode != 0:
        logger.error(f"Failed to create tag")
        exit(re.returncode)
    re = run(["git", "push", "--tags"], env=os.environ.copy())
    if re.returncode != 0:
        logger.error(f"Failed to push tags")
        exit(re.returncode)

    logger.info("Done.")
