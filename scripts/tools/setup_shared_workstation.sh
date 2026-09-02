#!/usr/bin/env bash
#
# Build a versioned, shared Isaac Sim/Isaac Lab/ROS 2 runtime without touching
# the workstation's existing environment.
#
# Stages:
#   sudo bash scripts/tools/setup_shared_workstation.sh prepare
#   bash scripts/tools/setup_shared_workstation.sh install
#   sudo bash scripts/tools/setup_shared_workstation.sh verify --accept-eula
#   sudo bash scripts/tools/setup_shared_workstation.sh activate --project-gates-passed
#   sudo bash scripts/tools/setup_shared_workstation.sh rollback
#
# Do not run "activate" until this repository passes its migration gates.

set -Eeuo pipefail

readonly GROUP_NAME="isaac_ros2"
readonly INSTALL_USER="acolab"
readonly PROJECT_USERS=(acolab marksu jychen)
readonly ISAAC_SIM_VERSION="6.0.1.0"
readonly ISAAC_LAB_BRANCH="release/3.0.0-beta2"
readonly TORCH_VERSION="2.10.0"
readonly TORCHVISION_VERSION="0.25.0"
readonly FASTAPI_VERSION="0.120.4"
readonly UVICORN_VERSION="0.29.0"
readonly WEBSOCKETS_VERSION="12.0"
readonly RELEASE_ID="isaacsim-6.0.1.0_isaaclab-3.0.0-beta2"
readonly SHARED_ROOT="/opt/isaac_ros2"
readonly RELEASE_ROOT="${SHARED_ROOT}/releases/${RELEASE_ID}"
readonly SHARED_DATA="/srv/isaac_ros2"
readonly STATE_ROOT="/var/lib/isaac_ros2-workstation"
readonly STATE_DIR="${STATE_ROOT}/${RELEASE_ID}"
readonly GLOBAL_STATE_DIR="${STATE_ROOT}/shared"
readonly DATA_MARKER="${SHARED_DATA}/.managed-by-isaac-ros2"
readonly MARKER="${RELEASE_ROOT}/.managed-by-isaac-ros2"
readonly ROS_SOURCE="/home/acolab/ros2_jazzy_ws/src"

log() {
    printf '[isaac_ros2] %s\n' "$*"
}

die() {
    printf '[isaac_ros2] ERROR: %s\n' "$*" >&2
    exit 1
}

require_root() {
    [[ "${EUID}" -eq 0 ]] || die "run this stage with sudo"
}

require_install_user() {
    [[ "${EUID}" -ne 0 ]] || die "do not run the install stage as root"
    [[ "$(id -un)" == "${INSTALL_USER}" ]] || die "the install stage must run as ${INSTALL_USER}"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command is missing: $1"
}

group_has_user() {
    local user="$1"
    id -nG "${user}" | tr ' ' '\n' | grep -Fxq "${GROUP_NAME}"
}

candidate_exists() {
    [[ -f "${MARKER}" ]] && [[ "$(<"${MARKER}")" == "${RELEASE_ID}" ]]
}

source_ros() {
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
    set -u
}

record_current_link() {
    if [[ -L "${SHARED_ROOT}/current" ]]; then
        readlink "${SHARED_ROOT}/current" >"${STATE_DIR}/current.before"
    elif [[ -e "${SHARED_ROOT}/current" ]]; then
        die "${SHARED_ROOT}/current exists and is not a symlink"
    else
        : >"${STATE_DIR}/current.absent"
    fi
}

rollback_on_exit() {
    local status="$?"
    trap - EXIT
    if [[ "${status}" -ne 0 ]]; then
        rollback_internal "prepare failed" || log "automatic rollback was incomplete; state was retained"
    fi
    exit "${status}"
}

prepare() {
    local user managed_path

    require_root
    for command in git rsync setfacl python3.12 colcon nvidia-smi; do
        require_command "${command}"
    done
    [[ -f /opt/ros/jazzy/setup.bash ]] || die "ROS 2 Jazzy is missing"
    [[ -d "${ROS_SOURCE}" ]] || die "ROS overlay source is missing: ${ROS_SOURCE}"
    for managed_path in "${STATE_ROOT}" "${GLOBAL_STATE_DIR}" "${SHARED_ROOT}" "${SHARED_ROOT}/releases"; do
        [[ ! -L "${managed_path}" ]] || die "managed path must not be a symlink: ${managed_path}"
    done
    [[ ! -e "${STATE_DIR}" && ! -L "${STATE_DIR}" ]] || die "state already exists; use status or rollback"
    [[ ! -e "${RELEASE_ROOT}" && ! -L "${RELEASE_ROOT}" ]] || die "candidate already exists: ${RELEASE_ROOT}"
    if [[ -e "${SHARED_ROOT}/current" && ! -L "${SHARED_ROOT}/current" ]]; then
        die "${SHARED_ROOT}/current exists and is not a symlink"
    fi
    if [[ -L "${SHARED_DATA}" ]]; then
        die "${SHARED_DATA} must not be a symlink"
    fi
    if [[ -e "${SHARED_DATA}" ]] && { [[ ! -f "${DATA_MARKER}" ]] || [[ "$(<"${DATA_MARKER}")" != "${GROUP_NAME}" ]]; }; then
        die "${SHARED_DATA} exists but is not managed by this tool"
    fi

    for user in "${PROJECT_USERS[@]}"; do
        id "${user}" >/dev/null 2>&1 || die "Linux account does not exist: ${user}"
    done

    install -d -o root -g root -m 0700 "${STATE_ROOT}"
    install -d -o root -g root -m 0700 "${GLOBAL_STATE_DIR}"
    install -d -o root -g root -m 0700 "${STATE_DIR}"
    trap rollback_on_exit EXIT

    if ! getent group "${GROUP_NAME}" >/dev/null; then
        : >"${GLOBAL_STATE_DIR}/group.created"
        groupadd "${GROUP_NAME}"
    fi

    for user in "${PROJECT_USERS[@]}"; do
        if ! group_has_user "${user}"; then
            : >"${GLOBAL_STATE_DIR}/member.${user}.added"
            usermod -aG "${GROUP_NAME}" "${user}"
        fi
    done

    install -d -o root -g "${GROUP_NAME}" -m 0755 "${SHARED_ROOT}"
    install -d -o root -g "${GROUP_NAME}" -m 0755 "${SHARED_ROOT}/releases"
    record_current_link
    if [[ -f "${SHARED_ROOT}/setup.bash" ]]; then
        cp --preserve=mode,ownership,timestamps "${SHARED_ROOT}/setup.bash" "${STATE_DIR}/setup.before"
    elif [[ -e "${SHARED_ROOT}/setup.bash" || -L "${SHARED_ROOT}/setup.bash" ]]; then
        die "${SHARED_ROOT}/setup.bash exists and is not a regular file"
    else
        : >"${STATE_DIR}/setup.absent"
    fi
    : >"${STATE_DIR}/release.created"
    install -d -o "${INSTALL_USER}" -g "${GROUP_NAME}" -m 2750 "${RELEASE_ROOT}"
    printf '%s\n' "${RELEASE_ID}" >"${MARKER}"
    chown "${INSTALL_USER}:${GROUP_NAME}" "${MARKER}"
    chmod 0640 "${MARKER}"

    if [[ ! -d "${SHARED_DATA}" ]]; then
        : >"${GLOBAL_STATE_DIR}/data.created"
        install -d -o root -g "${GROUP_NAME}" -m 2775 "${SHARED_DATA}"
        install -d -o root -g "${GROUP_NAME}" -m 2775 "${SHARED_DATA}/datasets" "${SHARED_DATA}/models" "${SHARED_DATA}/assets" "${SHARED_DATA}/outputs"
        printf '%s\n' "${GROUP_NAME}" >"${DATA_MARKER}"
        chown root:"${GROUP_NAME}" "${DATA_MARKER}"
        chmod 0644 "${DATA_MARKER}"
        setfacl -m "g:${GROUP_NAME}:rwx,m::rwx" "${SHARED_DATA}"
        setfacl -d -m "g:${GROUP_NAME}:rwx,m::rwx" "${SHARED_DATA}"
        setfacl -m "g:${GROUP_NAME}:rwx,m::rwx" "${SHARED_DATA}"/*
        setfacl -d -m "g:${GROUP_NAME}:rwx,m::rwx" "${SHARED_DATA}"/*
    else
        for directory in datasets models assets outputs; do
            [[ -d "${SHARED_DATA}/${directory}" ]] || die "managed shared data is incomplete: ${directory}"
        done
    fi

    : >"${STATE_DIR}/prepared"
    trap - EXIT
    log "candidate directory prepared: ${RELEASE_ROOT}"
    log "users must start a new login session before cross-account verification"
    log "next: bash $0 install"
}

write_setup() {
    local setup_file="${RELEASE_ROOT}/setup.bash"

    {
        printf '%s\n' '# Source this file; do not execute it.'
        printf 'export ISAAC_ROS2_RELEASE=%q\n' "${RELEASE_ROOT}"
        printf 'export ISAACLAB_ROOT=%q\n' "${RELEASE_ROOT}/IsaacLab"
        printf 'export ISAAC_ROS2_DATA=%q\n' "${SHARED_DATA}"
        cat <<'SETUP'
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "source this file instead of executing it" >&2
    exit 1
fi
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    echo "run 'conda deactivate' before sourcing this environment" >&2
    return 1
fi
if [[ -n "${VIRTUAL_ENV:-}" && "${VIRTUAL_ENV}" != "${ISAAC_ROS2_RELEASE}/env" ]]; then
    echo "deactivate the current Python venv before sourcing this environment" >&2
    return 1
fi

_isaac_ros2_nounset=false
[[ "$-" == *u* ]] && _isaac_ros2_nounset=true
set +u
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
export ISAAC_ROS2_PORTABLE_ROOT="${XDG_CACHE_HOME}/isaac_ros2/isaacsim-6.0.1.0_isaaclab-3.0.0-beta2/kit"
export ISAAC_ROS2_KIT_DATA_DIR="${ISAAC_ROS2_PORTABLE_ROOT}/data"
export ISAAC_ROS2_KIT_CACHE_DIR="${ISAAC_ROS2_PORTABLE_ROOT}/cache"
export ISAAC_ROS2_KIT_LOG_DIR="${ISAAC_ROS2_PORTABLE_ROOT}/logs"
export ISAAC_ROS2_KIT_CRASH_DIR="${ISAAC_ROS2_PORTABLE_ROOT}/crash"
export ISAAC_ROS2_SHADER_CACHE_DIR="${ISAAC_ROS2_KIT_CACHE_DIR}/shadercache"
export ISAAC_ROS2_DRIVER_SHADER_CACHE_DIR="${ISAAC_ROS2_KIT_CACHE_DIR}/nv_shadercache"
export WARP_CACHE_PATH="${XDG_CACHE_HOME}/isaac_ros2/isaacsim-6.0.1.0_isaaclab-3.0.0-beta2/warp"
export PYTHONPYCACHEPREFIX="${XDG_CACHE_HOME}/isaac_ros2/isaacsim-6.0.1.0_isaaclab-3.0.0-beta2/pycache"
mkdir -p "${ISAAC_ROS2_KIT_DATA_DIR}/exts" "${ISAAC_ROS2_KIT_CACHE_DIR}/DerivedDataCache" "${ISAAC_ROS2_SHADER_CACHE_DIR}" "${ISAAC_ROS2_DRIVER_SHADER_CACHE_DIR}" "${ISAAC_ROS2_KIT_LOG_DIR}" "${ISAAC_ROS2_KIT_CRASH_DIR}" "${WARP_CACHE_PATH}" "${PYTHONPYCACHEPREFIX}"

source /opt/ros/jazzy/setup.bash
if [[ -f "${ISAAC_ROS2_RELEASE}/ros2_ws/install/setup.bash" ]]; then
    source "${ISAAC_ROS2_RELEASE}/ros2_ws/install/setup.bash"
fi
source "${ISAAC_ROS2_RELEASE}/env/bin/activate"
hash -r

# The venv lives in a generic internal directory named "env".  Give every
# shared-workstation shell an unambiguous prompt without changing that stable
# filesystem layout.  The activation script still owns restoration on
# `deactivate` through its saved _OLD_VIRTUAL_PS1 value.
_isaac_ros2_previous_prompt="${VIRTUAL_ENV_PROMPT:-}"
if [[ -z "${VIRTUAL_ENV_DISABLE_PROMPT:-}" && -n "${_isaac_ros2_previous_prompt}" ]]; then
    PS1="(isaac_ros2) ${PS1#\("${_isaac_ros2_previous_prompt}"\) }"
    export PS1
fi
export VIRTUAL_ENV_PROMPT=isaac_ros2
unset _isaac_ros2_previous_prompt

if [[ "${_isaac_ros2_nounset}" == true ]]; then
    set -u
fi
unset _isaac_ros2_nounset

isaaclab() {
    if [[ "$#" -lt 1 ]]; then
        echo "usage: isaaclab SCRIPT.py [arguments...]" >&2
        return 2
    fi
    "${ISAACLAB_ROOT}/isaaclab.sh" -p "$@" --kit_args "--portable --portable-root=${ISAAC_ROS2_PORTABLE_ROOT} --/app/userConfigPath=${ISAAC_ROS2_KIT_DATA_DIR}/user.config.json --/app/extensions/registryCache=${ISAAC_ROS2_KIT_DATA_DIR}/exts --/UJITSO/datastore/localCachePath=${ISAAC_ROS2_KIT_CACHE_DIR}/DerivedDataCache --/rtx/shaderDb/shaderCachePath=${ISAAC_ROS2_SHADER_CACHE_DIR} --/rtx/shaderDb/driverShaderCachePath=${ISAAC_ROS2_DRIVER_SHADER_CACHE_DIR} --/log/file=${ISAAC_ROS2_KIT_LOG_DIR}/kit.log --/crashreporter/dumpDir=${ISAAC_ROS2_KIT_CRASH_DIR}"
}
SETUP
    } >"${setup_file}"
    chmod 0640 "${setup_file}"
}

install_candidate() {
    local bootstrap="${RELEASE_ROOT}/tools"
    local env_root="${RELEASE_ROOT}/env"
    local isaaclab_root="${RELEASE_ROOT}/IsaacLab"
    local ros_ws="${RELEASE_ROOT}/ros2_ws"
    local uv

    require_install_user
    candidate_exists || die "run prepare first"
    [[ ! -f "${RELEASE_ROOT}/.installed" ]] || die "candidate is already installed"
    [[ ! -e "${env_root}" && ! -e "${isaaclab_root}" ]] || die "partial install found; run rollback before retrying"
    export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    export PYTHONNOUSERSITE=1
    export UV_HTTP_TIMEOUT=300
    export UV_HTTP_RETRIES=10
    export UV_CONCURRENT_DOWNLOADS=4
    unset VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONHOME PYTHONPATH LD_LIBRARY_PATH
    unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION

    require_command python3.12
    require_command git
    require_command rsync
    if (( $(df -Pk "${RELEASE_ROOT}" | awk 'NR==2 {print $4}') < 83886080 )); then
        die "less than 80 GiB is free on the candidate filesystem"
    fi

    log "bootstrapping uv inside the candidate (no shell profile changes)"
    python3.12 -m venv "${bootstrap}"
    "${bootstrap}/bin/python" -m pip install --upgrade pip uv
    uv="${bootstrap}/bin/uv"

    log "creating Python 3.12 environment"
    "${uv}" venv --python /usr/bin/python3.12 --seed --prompt "${GROUP_NAME}" "${env_root}"

    log "installing Isaac Sim ${ISAAC_SIM_VERSION}"
    "${uv}" pip install --python "${env_root}/bin/python" "isaacsim[all,extscache]==${ISAAC_SIM_VERSION}" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow

    log "installing the Isaac Lab beta2 PyTorch compatibility set"
    "${uv}" pip install --python "${env_root}/bin/python" --upgrade "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url https://download.pytorch.org/whl/cu128

    log "cloning Isaac Lab ${ISAAC_LAB_BRANCH}"
    git clone --single-branch --branch "${ISAAC_LAB_BRANCH}" https://github.com/isaac-sim/IsaacLab.git "${isaaclab_root}"

    log "installing Isaac Lab extensions"
    (
        export VIRTUAL_ENV="${env_root}"
        export PATH="${env_root}/bin:${PATH}"
        export PYTHONNOUSERSITE=1
        cd "${isaaclab_root}"
        ./isaaclab.sh --install
    )

    log "installing this project's non-GUI runtime dependencies"
    # Keep the bridge stack aligned with isaacsim-kernel.  In particular,
    # Uvicorn 0.29 uses the legacy websockets protocol that Isaac Sim pins to
    # 12.0; installing an unconstrained newer release can break Ping/Pong while
    # a stream is connected.
    "${uv}" pip install --python "${env_root}/bin/python" \
        "fastapi==${FASTAPI_VERSION}" \
        "uvicorn==${UVICORN_VERSION}" \
        "websockets==${WEBSOCKETS_VERSION}" \
        requests pillow h5py opencv-python-headless

    log "copying and rebuilding the small ROS overlay in a neutral path"
    install -d "${ros_ws}/src"
    rsync -a --exclude '__pycache__' --exclude '.pytest_cache' "${ROS_SOURCE}/" "${ros_ws}/src/"
    (
        source_ros
        cd "${ros_ws}"
        colcon build --merge-install --cmake-args -DCMAKE_BUILD_TYPE=Release
    )

    write_setup
    install -d "${RELEASE_ROOT}/manifest"
    git -C "${isaaclab_root}" rev-parse HEAD >"${RELEASE_ROOT}/manifest/isaaclab-commit.txt"
    "${uv}" pip freeze --python "${env_root}/bin/python" >"${RELEASE_ROOT}/manifest/python-freeze.txt"
    "${uv}" --version >"${RELEASE_ROOT}/manifest/uv-version.txt"
    nvidia-smi --query-gpu=driver_version,name --format=csv,noheader >"${RELEASE_ROOT}/manifest/gpu.txt"
    find "${ros_ws}/src" -type f -print0 | sort -z | xargs -0 sha256sum >"${RELEASE_ROOT}/manifest/ros-source-sha256.txt"
    printf '%s\n' "${RELEASE_ID}" >"${RELEASE_ROOT}/.installed"
    chgrp -R "${GROUP_NAME}" "${RELEASE_ROOT}"
    chmod -R u+rwX,g+rX,g-w,o-rwx "${RELEASE_ROOT}"
    log "candidate installation complete; it is not active"
    log "next: sudo bash $0 verify --accept-eula"
}

verify_one_user() {
    local user="$1"
    local home
    home="$(getent passwd "${user}" | cut -d: -f6)"
    [[ -n "${home}" && -d "${home}" ]] || die "invalid home for ${user}"

    log "checking imports, CUDA, ROS, and headless Isaac as ${user}"
    runuser -u "${user}" -- env -i HOME="${home}" USER="${user}" LOGNAME="${user}" SHELL=/bin/bash PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin XDG_CACHE_HOME="${home}/.cache" HEADLESS=1 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES /bin/bash --noprofile --norc -c '
            set -eo pipefail
            source "$1"
            python3 -c '"'"'
import importlib.metadata as md
import os
import sys
import isaaclab
import rclpy
import torch

root = os.environ["ISAAC_ROS2_RELEASE"]
assert sys.executable.startswith(root + "/env/"), sys.executable
assert isaaclab.__file__.startswith(root + "/IsaacLab/"), isaaclab.__file__
assert rclpy.__file__.startswith("/opt/ros/jazzy/"), rclpy.__file__
assert md.version("isaacsim") == "6.0.1.0"
assert md.version("fastapi") == "0.120.4"
assert md.version("uvicorn") == "0.29.0"
assert md.version("websockets") == "12.0"
assert torch.__version__.startswith("2.10.0")
assert torch.cuda.is_available()
print("Python:", sys.executable)
print("Isaac Lab:", isaaclab.__file__)
print("CUDA:", torch.cuda.get_device_name(0))
'"'"'
            test "$(ros2 pkg prefix vla_ros_bridge)" = "${ISAAC_ROS2_RELEASE}/ros2_ws/install"
            verify_sentinel="$(mktemp "${XDG_CACHE_HOME}/isaac-empty-scene.XXXXXX")"
            rm -f "${verify_sentinel}"
            export ISAAC_ROS2_VERIFY_SENTINEL="${verify_sentinel}"
            trap '"'"'rm -f "${verify_sentinel}"'"'"' EXIT
            rm -f "${ISAAC_ROS2_KIT_LOG_DIR}/verify.log"
            timeout --kill-after=30s 300s python3 -c '"'"'
import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app
try:
    import carb
    from isaaclab.sim import SimulationCfg, SimulationContext
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors.camera import CameraCfg
    from isaaclab.utils.configclass import configclass

    settings = carb.settings.get_settings()
    assert settings.get("/rtx/shaderDb/shaderCachePath") == os.environ["ISAAC_ROS2_SHADER_CACHE_DIR"]
    assert settings.get("/rtx/shaderDb/driverShaderCachePath") == os.environ["ISAAC_ROS2_DRIVER_SHADER_CACHE_DIR"]
    settings.set_bool("/isaaclab/cameras_enabled", True)

    @configclass
    class VerifySceneCfg(InteractiveSceneCfg):
        camera = CameraCfg(
            prim_path="/World/VerifyCamera",
            update_period=0.0,
            height=64,
            width=64,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=21.0),
        )

    simulation = SimulationContext(SimulationCfg(dt=0.01))
    scene = InteractiveScene(VerifySceneCfg(num_envs=1, env_spacing=1.0))
    simulation.reset()
    simulation.step()
    scene.update(simulation.get_physics_dt())
    assert tuple(scene["camera"].data.output["rgb"].shape) == (1, 64, 64, 3)
    Path(os.environ["ISAAC_ROS2_VERIFY_SENTINEL"]).write_text("PASS\n")
    print("ISAAC_CAMERA_SCENE_PASS", flush=True)
finally:
    app.close()
'"'"' --enable_cameras --viz kit --kit_args "--portable --portable-root=${ISAAC_ROS2_PORTABLE_ROOT} --/app/userConfigPath=${ISAAC_ROS2_KIT_DATA_DIR}/user.config.json --/app/extensions/registryCache=${ISAAC_ROS2_KIT_DATA_DIR}/exts --/UJITSO/datastore/localCachePath=${ISAAC_ROS2_KIT_CACHE_DIR}/DerivedDataCache --/rtx/shaderDb/shaderCachePath=${ISAAC_ROS2_SHADER_CACHE_DIR} --/rtx/shaderDb/driverShaderCachePath=${ISAAC_ROS2_DRIVER_SHADER_CACHE_DIR} --/log/file=${ISAAC_ROS2_KIT_LOG_DIR}/verify.log --/crashreporter/dumpDir=${ISAAC_ROS2_KIT_CRASH_DIR}"
            test "$(cat "${verify_sentinel}")" = PASS || {
                echo "Isaac empty-scene verification did not complete reset and step" >&2
                exit 1
            }
            grep -Fq "${ISAAC_ROS2_KIT_CACHE_DIR}/DerivedDataCache" "${ISAAC_ROS2_KIT_LOG_DIR}/verify.log" || {
                echo "Isaac did not use the per-user DerivedDataCache" >&2
                exit 1
            }
            if grep -Fq "Failed to create local file data store" "${ISAAC_ROS2_KIT_LOG_DIR}/verify.log"; then
                echo "Isaac reported a DerivedDataCache creation failure" >&2
                exit 1
            fi
            rm -f "${verify_sentinel}"
            trap - EXIT
        ' _ "${RELEASE_ROOT}/setup.bash"
}

verify_candidate() {
    local gpu_status user

    require_root
    [[ "${2:-}" == "--accept-eula" ]] || die "verify requires --accept-eula (NVIDIA EULA acceptance)"
    candidate_exists || die "candidate marker is missing"
    [[ -f "${RELEASE_ROOT}/.installed" ]] || die "run install first"
    # A failed rerun must not leave a stale successful verification marker.
    rm -f "${RELEASE_ROOT}/.verified"

    if ! gpu_status="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1)"; then
        die "NVIDIA GPU preflight failed; recover the driver/GPU before verify: ${gpu_status}"
    fi
    [[ -n "${gpu_status}" ]] || die "NVIDIA GPU preflight found no devices"

    # Regenerate runtime setup so an interrupted/failed verification can safely
    # pick up installer fixes without reinstalling the large candidate.
    write_setup

    # pip creates files with the install user's primary group.  Normalize the
    # completed candidate before checking it from the other project accounts.
    chgrp -R "${GROUP_NAME}" "${RELEASE_ROOT}"
    chmod -R u+rwX,g+rX,g-w,o-rwx "${RELEASE_ROOT}"

    for user in "${PROJECT_USERS[@]}"; do
        group_has_user "${user}" || die "${user} is not in ${GROUP_NAME}"
        verify_one_user "${user}"
    done
    printf '%s\n' "$(date --iso-8601=seconds)" >"${RELEASE_ROOT}/.verified"
    chmod 0640 "${RELEASE_ROOT}/.verified"
    log "candidate runtime gate passed for all three users"
    log "project migration gates must still pass before activate"
}

current_matches_prepare() {
    if [[ -f "${STATE_DIR}/current.before" ]]; then
        [[ -L "${SHARED_ROOT}/current" ]] && [[ "$(readlink "${SHARED_ROOT}/current")" == "$(<"${STATE_DIR}/current.before")" ]]
    else
        [[ ! -e "${SHARED_ROOT}/current" && ! -L "${SHARED_ROOT}/current" ]]
    fi
}

setup_matches_prepare() {
    if [[ -f "${STATE_DIR}/setup.before" ]]; then
        [[ -f "${SHARED_ROOT}/setup.bash" ]] && cmp -s "${STATE_DIR}/setup.before" "${SHARED_ROOT}/setup.bash"
    else
        [[ ! -e "${SHARED_ROOT}/setup.bash" && ! -L "${SHARED_ROOT}/setup.bash" ]]
    fi
}

activate_candidate() {
    local temporary_link temporary_setup

    require_root
    [[ "${2:-}" == "--project-gates-passed" ]] || die "activate requires --project-gates-passed"
    candidate_exists || die "candidate marker is missing"
    [[ -f "${RELEASE_ROOT}/.installed" && -f "${RELEASE_ROOT}/.verified" ]] || die "candidate install/verify gates are incomplete"
    current_matches_prepare || die "current changed after prepare; refusing to overwrite it"
    setup_matches_prepare || die "setup.bash changed after prepare; refusing to overwrite it"

    chown -R root:"${GROUP_NAME}" "${RELEASE_ROOT}"
    chmod -R u+rwX,g+rX,g-w,o-rwx "${RELEASE_ROOT}"

    temporary_setup="$(mktemp "${SHARED_ROOT}/.setup.XXXXXX")"
    temporary_link="${SHARED_ROOT}/.current.$$"
    {
        printf '%s\n' '# Managed by setup_shared_workstation.sh.'
        printf '%s\n' 'source /opt/isaac_ros2/current/setup.bash'
    } >"${temporary_setup}"
    chown root:"${GROUP_NAME}" "${temporary_setup}"
    chmod 0644 "${temporary_setup}"
    ln -s "${RELEASE_ROOT}" "${temporary_link}"

    if ! current_matches_prepare || ! setup_matches_prepare; then
        rm -f "${temporary_setup}" "${temporary_link}"
        die "current/setup changed during activation; nothing was switched"
    fi
    if ! mv -Tf "${temporary_link}" "${SHARED_ROOT}/current"; then
        rm -f "${temporary_setup}" "${temporary_link}"
        die "failed to switch current; setup.bash was not changed"
    fi
    if ! mv -Tf "${temporary_setup}" "${SHARED_ROOT}/setup.bash"; then
        rm -f "${temporary_setup}"
        if restore_current_link; then
            die "failed to install setup.bash; previous current was restored"
        fi
        die "failed to install setup.bash and restore current; rollback state was retained"
    fi

    : >"${STATE_DIR}/activated"
    log "active runtime is now ${RELEASE_ROOT}"
}

shared_data_has_content() {
    local entry
    [[ -d "${SHARED_DATA}" ]] || return 1
    while IFS= read -r entry; do
        case "${entry}" in
            "${DATA_MARKER}"|"${SHARED_DATA}/datasets"|"${SHARED_DATA}/models"|"${SHARED_DATA}/assets"|"${SHARED_DATA}/outputs")
                ;;
            *)
                return 0
                ;;
        esac
    done < <(find "${SHARED_DATA}" -mindepth 1 -print)
    return 1
}

other_releases_exist() {
    find "${SHARED_ROOT}/releases" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null | grep -q .
}

setup_is_managed_entrypoint() {
    [[ -f "${SHARED_ROOT}/setup.bash" ]] && grep -Fqx '# Managed by setup_shared_workstation.sh.' "${SHARED_ROOT}/setup.bash"
}

restore_setup_file() {
    local temporary

    setup_matches_prepare && return 0
    if [[ -e "${SHARED_ROOT}/setup.bash" || -L "${SHARED_ROOT}/setup.bash" ]] && ! setup_is_managed_entrypoint; then
        log "ERROR: setup.bash changed externally; leaving it unchanged"
        return 1
    fi
    if [[ -f "${STATE_DIR}/setup.before" ]]; then
        temporary="${SHARED_ROOT}/.setup.rollback.$$"
        if ! cp --preserve=mode,ownership,timestamps "${STATE_DIR}/setup.before" "${temporary}"; then
            rm -f -- "${temporary}"
            return 1
        fi
        if ! mv -Tf -- "${temporary}" "${SHARED_ROOT}/setup.bash"; then
            rm -f -- "${temporary}"
            return 1
        fi
    elif [[ -f "${STATE_DIR}/setup.absent" ]]; then
        rm -f -- "${SHARED_ROOT}/setup.bash" || return 1
    else
        log "ERROR: setup.bash rollback record is missing"
        return 1
    fi
}

restore_current_link() {
    local current_target previous temporary

    current_matches_prepare && return 0
    if [[ ! -L "${SHARED_ROOT}/current" ]]; then
        log "ERROR: current changed externally; leaving it unchanged"
        return 1
    fi
    if ! current_target="$(readlink -f -- "${SHARED_ROOT}/current")"; then
        log "ERROR: cannot resolve current; leaving it unchanged"
        return 1
    fi
    if [[ "${current_target}" != "${RELEASE_ROOT}" ]]; then
        log "ERROR: current points elsewhere; leaving it unchanged: ${current_target}"
        return 1
    fi
    if [[ -f "${STATE_DIR}/current.before" ]]; then
        previous="$(<"${STATE_DIR}/current.before")"
        temporary="${SHARED_ROOT}/.current.rollback.$$"
        if ! ln -s -- "${previous}" "${temporary}"; then
            return 1
        fi
        if ! mv -Tf -- "${temporary}" "${SHARED_ROOT}/current"; then
            rm -f -- "${temporary}"
            return 1
        fi
    else
        rm -f -- "${SHARED_ROOT}/current" || return 1
    fi
}

restore_active_entrypoint() {
    local current_target

    if current_matches_prepare; then
        if setup_matches_prepare; then
            return 0
        fi
        if setup_is_managed_entrypoint; then
            restore_setup_file
            return $?
        fi
        log "ERROR: setup.bash changed externally; leaving setup and candidate unchanged"
        return 1
    fi
    if [[ ! -L "${SHARED_ROOT}/current" ]]; then
        log "ERROR: current changed externally; leaving current/setup and candidate unchanged"
        return 1
    fi
    if ! current_target="$(readlink -f -- "${SHARED_ROOT}/current")"; then
        log "ERROR: cannot resolve current; leaving current/setup unchanged"
        return 1
    fi
    if [[ "${current_target}" != "${RELEASE_ROOT}" ]]; then
        log "current points elsewhere; leaving current/setup unchanged: ${current_target}"
        return 0
    fi
    restore_current_link || return 1
    restore_setup_file || return 1
}

remove_managed_shared_data() {
    local directory failed=false

    if [[ -L "${SHARED_DATA}" ]]; then
        log "ERROR: shared data changed into a symlink; refusing to remove it"
        return 1
    fi
    if [[ -e "${DATA_MARKER}" ]] && ! rm -f -- "${DATA_MARKER}"; then
        failed=true
    fi
    for directory in datasets models assets outputs; do
        if [[ -e "${SHARED_DATA}/${directory}" || -L "${SHARED_DATA}/${directory}" ]]; then
            rmdir -- "${SHARED_DATA}/${directory}" || failed=true
        fi
    done
    if [[ -e "${SHARED_DATA}" ]] && ! rmdir -- "${SHARED_DATA}"; then
        failed=true
    fi
    [[ "${failed}" == false ]]
}

remove_managed_group() {
    local user marker members failed=false

    for user in "${PROJECT_USERS[@]}"; do
        marker="${GLOBAL_STATE_DIR}/member.${user}.added"
        [[ -f "${marker}" ]] || continue
        if getent group "${GROUP_NAME}" >/dev/null && group_has_user "${user}"; then
            if ! gpasswd -d "${user}" "${GROUP_NAME}" >/dev/null 2>&1; then
                failed=true
                continue
            fi
        fi
        rm -f -- "${marker}" || failed=true
    done
    if [[ -f "${GLOBAL_STATE_DIR}/group.created" ]]; then
        if getent group "${GROUP_NAME}" >/dev/null; then
            members="$(getent group "${GROUP_NAME}" | cut -d: -f4)"
            if [[ "${failed}" == true || -n "${members}" ]]; then
                [[ -z "${members}" ]] || log "ERROR: preserving ${GROUP_NAME}; it has unmanaged members: ${members}"
                failed=true
            elif groupdel "${GROUP_NAME}" >/dev/null 2>&1; then
                rm -f -- "${GLOBAL_STATE_DIR}/group.created" || failed=true
            else
                failed=true
            fi
        else
            rm -f -- "${GLOBAL_STATE_DIR}/group.created" || failed=true
        fi
    fi
    [[ "${failed}" == false ]]
}

rollback_internal() {
    local reason="${1:-requested}"
    local failed=false entrypoint_ok=true keep_group=false cleanup_deferred=false data_present=false releases_present=false

    trap - ERR
    set +e
    log "rollback: ${reason}"
    if [[ ! -d "${STATE_DIR}" ]]; then
        log "no managed candidate exists"
        return 0
    fi

    if ! restore_active_entrypoint; then
        failed=true
        entrypoint_ok=false
    fi
    if [[ -f "${STATE_DIR}/release.created" && ( -e "${RELEASE_ROOT}" || -L "${RELEASE_ROOT}" ) ]]; then
        if [[ "${entrypoint_ok}" == false ]]; then
            log "ERROR: retaining candidate because its active entrypoint could not be restored"
        elif [[ -L "${RELEASE_ROOT}" ]]; then
            log "ERROR: candidate changed into a symlink; refusing to remove it"
            failed=true
        else
            case "${RELEASE_ROOT}" in
                "${SHARED_ROOT}/releases/"*)
                    if ! rm -rf --one-file-system -- "${RELEASE_ROOT}" || [[ -e "${RELEASE_ROOT}" || -L "${RELEASE_ROOT}" ]]; then
                        failed=true
                    fi
                    ;;
                *)
                    log "ERROR: refusing unsafe release path: ${RELEASE_ROOT}"
                    failed=true
                    ;;
            esac
        fi
    fi

    shared_data_has_content && data_present=true
    other_releases_exist && releases_present=true
    if [[ "${data_present}" == true || "${releases_present}" == true ]]; then
        keep_group=true
        log "preserving shared data/group because data or another release exists"
        if [[ "${data_present}" == true && "${releases_present}" == false ]]; then
            cleanup_deferred=true
        fi
    elif [[ -f "${GLOBAL_STATE_DIR}/data.created" ]]; then
        if remove_managed_shared_data; then
            rm -f -- "${GLOBAL_STATE_DIR}/data.created" || failed=true
        else
            failed=true
        fi
    elif [[ -e "${SHARED_DATA}" || -L "${SHARED_DATA}" ]]; then
        keep_group=true
        log "preserving pre-existing shared data/group"
    fi

    if [[ "${failed}" == true ]]; then
        keep_group=true
    fi
    if [[ "${keep_group}" == false ]] && ! remove_managed_group; then
        failed=true
    fi

    if [[ "${failed}" == true ]]; then
        log "ERROR: rollback incomplete; state retained at ${STATE_DIR}"
        return 1
    fi
    if [[ "${cleanup_deferred}" == true ]]; then
        log "ERROR: rollback deferred because shared data remains; state retained at ${STATE_DIR}"
        return 1
    fi
    if ! rm -rf --one-file-system -- "${STATE_DIR}" || [[ -e "${STATE_DIR}" || -L "${STATE_DIR}" ]]; then
        log "ERROR: rollback cleanup failed; inspect ${STATE_DIR}"
        return 1
    fi
    rmdir "${SHARED_ROOT}/releases" "${SHARED_ROOT}" 2>/dev/null || true
    rmdir "${GLOBAL_STATE_DIR}" "${STATE_ROOT}" 2>/dev/null || true
    log "rollback complete; the original /home/acolab environment was untouched"
}

status() {
    printf 'group:      '
    getent group "${GROUP_NAME}" || printf 'not created\n'
    printf 'candidate:  %s\n' "${RELEASE_ROOT}"
    printf 'prepared:   %s\n' "$(candidate_exists && echo yes || echo no)"
    printf 'installed:  %s\n' "$(test -f "${RELEASE_ROOT}/.installed" && echo yes || echo no)"
    printf 'verified:   %s\n' "$(test -f "${RELEASE_ROOT}/.verified" && echo yes || echo no)"
    printf 'current:    %s\n' "$(readlink "${SHARED_ROOT}/current" 2>/dev/null || echo not-active)"
}

main() {
    case "${1:-}" in
        prepare) prepare ;;
        install) install_candidate ;;
        verify) verify_candidate "$@" ;;
        activate) activate_candidate "$@" ;;
        rollback)
            require_root
            rollback_internal "requested by user"
            ;;
        status) status ;;
        *)
            die "usage: $0 {prepare|install|verify --accept-eula|activate --project-gates-passed|rollback|status}"
            ;;
    esac
}

main "$@"
