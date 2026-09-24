"""Verify the maintained inference boundary on the allocated compute host."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.scripts.inference_endpoint_security import EndpointSecurityError
from analysis.scripts.inference_network_namespace import (
    MARKER,
    _verify_exclusive_slurm_boundary,
)


def cluster_profile(cluster: str) -> dict:
    if cluster not in {"jupiter", "horeka"}:
        raise EndpointSecurityError("Unknown execution cluster")
    path = Path(__file__).resolve().parents[1] / "config/cluster_execution" / f"{cluster}.json"
    return json.loads(path.read_bytes())


def verify(cluster: str, boundary: str | None = None, *, profile: dict | None = None) -> dict:
    profile = profile or cluster_profile(cluster)
    boundary = boundary or profile.get("execution_boundary")
    if profile.get("cluster") != cluster:
        raise EndpointSecurityError("Execution boundary profile belongs to a different cluster")
    if cluster not in {"jupiter", "horeka"} or boundary != "exclusive-slurm-node":
        raise EndpointSecurityError("Select an explicit supported cluster execution boundary")
    # This selection deliberately overrides inherited terminal/site settings.
    os.environ["GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY"] = "1"
    os.environ.pop(MARKER, None)
    receipt = _verify_exclusive_slurm_boundary(
        allow_jupiter_full_node=cluster == "jupiter" and profile.get("allow_jupiter_full_node_representation", True),
    )
    try:
        hosts = subprocess.check_output(
            ["scontrol", "show", "hostnames", receipt["node_list"]], text=True, timeout=10,
        ).splitlines()
    except (OSError, subprocess.SubprocessError) as error:
        raise EndpointSecurityError("Cannot verify allocated compute-host membership") from error
    if socket.gethostname().split(".")[0] not in {host.split(".")[0] for host in hosts}:
        raise EndpointSecurityError("Boundary verification must run on the allocated compute node, not a login host")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", required=True, choices=["jupiter", "horeka"])
    parser.add_argument("--boundary", choices=["exclusive-slurm-node", "private-network-namespace"])
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--exec", dest="command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        profile = json.loads(args.profile.read_bytes()) if args.profile else cluster_profile(args.cluster)
        selected = args.boundary or profile.get("execution_boundary")
        if selected == "private-network-namespace":
            if not args.command:
                raise EndpointSecurityError("Explicit namespace backend requires --exec; a child check cannot isolate its parent")
            from analysis.scripts.inference_network_namespace import (
                ensure_private_network_namespace,
            )
            os.environ["GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY"] = "0"
            receipt = ensure_private_network_namespace(args.command)
        else:
            receipt = verify(args.cluster, selected, profile=profile)
        print("INFERENCE_ALLOCATION_BOUNDARY=" + json.dumps(receipt, sort_keys=True), flush=True)
        if args.command:
            os.execvp(args.command[0], args.command)
        return 0
    except EndpointSecurityError as error:
        print(f"INFERENCE_ALLOCATION_BOUNDARY=FAIL {error}; no namespace fallback", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
