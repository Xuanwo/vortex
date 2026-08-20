#!/usr/bin/env bash

set -Eeuo pipefail

if [[ ${EUID} -ne 0 ]]; then
    echo "Run this script as root." >&2
    exit 1
fi

dnf install -y \
    clang cmake curl gcc gcc-c++ git jq mdadm ninja-build numactl \
    openssl-devel pkgconf-pkg-config protobuf-compiler protobuf-devel \
    python3 python3-pip time unzip xfsprogs zstd

if ! command -v duckdb >/dev/null 2>&1; then
    curl -fsSL \
        https://github.com/duckdb/duckdb/releases/download/v1.5.5/duckdb_cli-linux-amd64.zip \
        -o /tmp/duckdb-cli.zip
    unzip -p /tmp/duckdb-cli.zip duckdb >/usr/local/bin/duckdb
    chmod 0755 /usr/local/bin/duckdb
fi

mount_point=/mnt/bench
if mountpoint -q "${mount_point}"; then
    exit 0
fi

mapfile -t instance_store_devices < <(
    lsblk -dn -o PATH,MODEL | awk '/Amazon EC2 NVMe Instance Storage/ {print $1}'
)

if [[ ${#instance_store_devices[@]} -eq 0 ]]; then
    echo "No EC2 instance-store NVMe devices found." >&2
    exit 1
fi

if [[ ${#instance_store_devices[@]} -eq 1 ]]; then
    data_device=${instance_store_devices[0]}
else
    mdadm --create /dev/md0 --level=0 --raid-devices="${#instance_store_devices[@]}" \
        --force "${instance_store_devices[@]}"
    udevadm settle
    data_device=/dev/md0
fi

mkfs.xfs -f "${data_device}"
mkdir -p "${mount_point}"
mount -o noatime "${data_device}" "${mount_point}"
chown ec2-user:ec2-user "${mount_point}"

lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS,MODEL
df -h "${mount_point}"
