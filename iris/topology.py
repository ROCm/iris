# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import json
import logging
import os
import re
import socket
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger("iris.topology")


class InterconnectLevel(IntEnum):
    """Hierarchical interconnect tiers."""

    INTRA_NODE = 0
    INTRA_RACK_FABRIC = 1
    INTER_NODE_RDMA = 2

    def __str__(self) -> str:
        return self.name


class IntraNodeLinkType(IntEnum):
    """Link type between two GPUs."""

    SELF = -1  # Same GPU (diagonal)
    NVLINK = 0  # NVIDIA NVLink or AMD xGMI
    NVSWITCH = 1  # Connected through NVSwitch (all-to-all NVLink)
    PCIE_SWITCH = 2  # Same PCIe switch (PIX/PXB)
    PCIE_HOST_BRIDGE = 3  # Same CPU socket / PCIe host bridge (PHB)
    PCIE_NUMA = 4  # Crosses NUMA boundary (NODE)
    PCIE_SYSTEM = 5  # Crosses QPI/UPI between sockets (SYS)
    UNKNOWN = 99

    def __str__(self) -> str:
        return self.name


@dataclass
class FabricInfo:
    """
    GPU fabric domain identification.

    This is a vendor-agnostic representation of where a GPU sits in the
    physical fabric topology. Two GPUs with the same (cluster_uuid, clique_id)
    are in the same high-speed fabric domain and can communicate via direct
    GPU-to-GPU links (NVLink via NVSwitch, or xGMI) without RDMA.

    AMD mapping:
        cluster_uuid  <->  ppod_id   (physical pod identifier, uint64)
        clique_id     <->  vpod_id   (virtual pod identifier, uint32)

    NVIDIA mapping:
        cluster_uuid  <->  clusterUuid[16]  (NVLink domain UUID, bytes)
        clique_id     <->  cliqueId         (fabric clique ID, uint32)

    If both fields are empty/zero, fabric info is unavailable (e.g. no
    NVSwitch, no xGMI hive, single-node PCIe-only system).
    """

    cluster_uuid: str = ""  # Domain identifier (ppod_id hex / clusterUuid hex)
    clique_id: int = 0  # Sub-domain identifier (vpod_id / cliqueId)

    @property
    def is_valid(self) -> bool:
        """True if fabric info was successfully retrieved."""
        return bool(self.cluster_uuid)

    @property
    def domain_key(self) -> str:
        """
        Combined key for domain comparison.

        Two GPUs with the same domain_key are in the same fabric domain.
        """
        if not self.is_valid:
            return ""
        return f"{self.cluster_uuid}:{self.clique_id}"

    def to_dict(self) -> dict:
        return {"cluster_uuid": self.cluster_uuid, "clique_id": self.clique_id}

    @classmethod
    def from_dict(cls, d: dict) -> "FabricInfo":
        return cls(cluster_uuid=d.get("cluster_uuid", ""), clique_id=d.get("clique_id", 0))


# ---------------------------------------------------------------------------
# Logical-to-physical GPU index translation
# ---------------------------------------------------------------------------


def _logical_to_physical_gpu_index(logical_idx: int, vendor: str) -> int:
    """
    Translate a logical (PyTorch) GPU index to the physical index used by
    vendor libraries (NVML / AMDSMI).

    PyTorch (CUDA runtime) respects CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES
    and remaps device indices so that logical 0 = first *visible* GPU.
    NVML and AMDSMI always enumerate *all* physical GPUs starting from 0.

    This function parses the relevant visibility env vars to recover the
    physical index.  If no env var is set or the entry is not a plain
    integer (e.g. GPU UUIDs in CUDA_VISIBLE_DEVICES), it falls back to
    returning the logical index unchanged — callers that need robustness
    against UUID-style entries should prefer PCI-bus-ID-based handle
    resolution instead (see _get_nvml_handle_by_pci / _get_amdsmi_handle_by_pci).
    """
    if logical_idx < 0:
        return logical_idx

    if vendor == "nvidia":
        env_vars = ["CUDA_VISIBLE_DEVICES"]
    elif vendor == "amd":
        # ROCm checks HIP_VISIBLE_DEVICES first, then ROCR_VISIBLE_DEVICES
        env_vars = ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"]
    else:
        return logical_idx

    for env_var in env_vars:
        visible = os.environ.get(env_var)
        if not visible:
            continue

        entries = [e.strip() for e in re.split(r"[,\s]+", visible) if e.strip()]
        if logical_idx >= len(entries):
            continue

        entry = entries[logical_idx]

        # Skip UUID-style entries (e.g. "GPU-xxxxxxxx-...") — can't map
        # to a plain integer index.  Caller should use PCI-based resolution.
        if not entry.isdigit() and not (entry.startswith("-") and entry[1:].isdigit()):
            logger.debug(
                "%s entry '%s' is not a plain index; falling back to logical index %d",
                env_var,
                entry,
                logical_idx,
            )
            return logical_idx

        try:
            return int(entry)
        except ValueError:
            return logical_idx

    return logical_idx


# ---------------------------------------------------------------------------
# PCI-bus-ID-based vendor handle resolution
# ---------------------------------------------------------------------------


def _get_nvml_handle_by_pci(pci_bus_id: str):
    """
    Get an NVML device handle by PCI bus ID.

    This bypasses index-based lookup entirely and is immune to
    CUDA_VISIBLE_DEVICES remapping.  Returns None on failure.
    """
    try:
        import pynvml
    except ImportError:
        return None

    norm = _normalize_pci_bus_id(pci_bus_id)
    # nvmlDeviceGetHandleByPciBusId expects full domain:bus:dev.fn
    if len(norm.split(":")) == 2:
        norm = f"0000:{norm}"

    try:
        return pynvml.nvmlDeviceGetHandleByPciBusId(norm.encode())
    except Exception as e:
        logger.debug("nvmlDeviceGetHandleByPciBusId(%s) failed: %s", norm, e)
        return None


def _get_amdsmi_handle_by_pci(pci_bus_id: str, all_handles=None):
    """
    Get an amdsmi processor handle by PCI bus ID (BDF).

    If *all_handles* is provided, it is used directly; otherwise
    amdsmi_get_processor_handles() is called (caller must have
    already called amdsmi_init).

    Returns None if no match is found.
    """
    try:
        import amdsmi
    except ImportError:
        return None

    if all_handles is None:
        all_handles = amdsmi.amdsmi_get_processor_handles()

    norm = _normalize_pci_bus_id(pci_bus_id)

    for handle in all_handles:
        try:
            bdf = amdsmi.amdsmi_get_gpu_device_bdf(handle)
            if bdf and _normalize_pci_bus_id(str(bdf)) == norm:
                return handle
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Fabric info
# ---------------------------------------------------------------------------


def _amd_get_gpu_fabric_info(gpu_id: int, pci_bus_id: str = "") -> FabricInfo:
    """
    Get GPU fabric info from AMD's AMDSMI library.

    GPUs with matching (ppod_id, vpod_id) are in the same fabric
    domain and can communicate via direct GPU links without RDMA.

    Args:
        gpu_id: Local GPU device index (0-based, logical).
        pci_bus_id: PCI bus ID for handle resolution (preferred over index).

    Returns:
        FabricInfo with cluster_uuid = hex(ppod_id), clique_id = vpod_id.
        Returns empty FabricInfo if the call fails or is not available.
    """
    # Default placeholder: no fabric info available (single-node behavior)
    return FabricInfo()


def _nvidia_get_gpu_fabric_info(gpu_id: int, pci_bus_id: str = "") -> FabricInfo:
    """
    Get GPU fabric info from NVIDIA's NVML library.

    When *pci_bus_id* is provided, the NVML handle is resolved via PCI
    address — immune to CUDA_VISIBLE_DEVICES remapping.  Falls back to
    physical-index-based lookup if PCI resolution fails.

    Args:
        gpu_id: Local GPU device index (0-based, logical).
        pci_bus_id: PCI bus ID for handle resolution (preferred).
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            # --- Resolve handle: prefer PCI bus ID, fall back to physical index ---
            handle = None
            if pci_bus_id and pci_bus_id != "unknown":
                handle = _get_nvml_handle_by_pci(pci_bus_id)

            if handle is None:
                physical_idx = _logical_to_physical_gpu_index(gpu_id, "nvidia")
                handle = pynvml.nvmlDeviceGetHandleByIndex(physical_idx)

            fabric_info = None
            try:
                info_struct = pynvml.c_nvmlGpuFabricInfo_v2_t()
                pynvml.nvmlDeviceGetGpuFabricInfoV(handle, info_struct)
                fabric_info = info_struct
            except (AttributeError, TypeError, pynvml.NVMLError):
                # GPU doesn't support fabric
                return FabricInfo()

            if fabric_info is None:
                return FabricInfo()

            # Check registration state — must be COMPLETED (value 3)
            state = getattr(fabric_info, "state", None)
            if state is not None and state != 3:
                return FabricInfo()

            # Check status — must be SUCCESS (value 0)
            status = getattr(fabric_info, "status", None)
            if status is not None and status != 0:
                return FabricInfo()

            # Extract clusterUuid
            cluster_uuid_raw = getattr(fabric_info, "clusterUuid", None)
            if cluster_uuid_raw is None:
                return FabricInfo()

            if isinstance(cluster_uuid_raw, bytes):
                cluster_uuid_hex = cluster_uuid_raw.hex()
            elif isinstance(cluster_uuid_raw, (list, tuple)):
                cluster_uuid_hex = bytes(cluster_uuid_raw).hex()
            else:
                cluster_uuid_hex = str(cluster_uuid_raw)

            if all(c == "0" for c in cluster_uuid_hex):
                return FabricInfo()

            clique_id = getattr(fabric_info, "cliqueId", 0)

            return FabricInfo(
                cluster_uuid=cluster_uuid_hex,
                clique_id=int(clique_id),
            )
        finally:
            pynvml.nvmlShutdown()

    except ImportError:
        logger.debug("pynvml not available, skipping NVML fabric info")
    except Exception as e:
        logger.debug("NVML fabric info query failed for GPU %d: %s", gpu_id, e)

    return FabricInfo()


def get_gpu_fabric_info(gpu_id: int, vendor: str, pci_bus_id: str = "") -> FabricInfo:
    """
    Get GPU fabric domain info for the given device.

    Dispatches to the appropriate vendor-specific implementation:
        AMD:    amdsmi_get_gpu_fabric_info (ppod_id, vpod_id)
        NVIDIA: nvmlDeviceGetGpuFabricInfoV (clusterUuid, cliqueId)

    Args:
        gpu_id: Local GPU device index.
        vendor: "amd" or "nvidia".
        pci_bus_id: PCI bus ID for PCI-based handle resolution (preferred).

    Returns:
        FabricInfo identifying the fabric domain this GPU belongs to.
    """
    if vendor == "amd":
        return _amd_get_gpu_fabric_info(gpu_id, pci_bus_id=pci_bus_id)
    elif vendor == "nvidia":
        return _nvidia_get_gpu_fabric_info(gpu_id, pci_bus_id=pci_bus_id)
    else:
        return FabricInfo()


def _normalize_pci_bus_id(bus_id: str) -> str:
    """
    Normalize a PCI bus ID to a canonical lowercase form for comparison.

    Handles formats like:
        "0000:41:00.0"          -> "0000:41:00.0"
        "00000000:41:00.0"      -> "0000:41:00.0"   (nvidia sometimes uses 8-char domain)
        "GPU 0000:41:00.0"      -> "0000:41:00.0"   (strip prefix junk)
    """
    bus_id = bus_id.strip().lower()
    # Extract the BDF pattern (domain:bus:device.function)
    match = re.search(r"([0-9a-f]+:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f])", bus_id)
    if not match:
        return bus_id
    bdf = match.group(1)
    # Normalize domain to 4 hex chars
    parts = bdf.split(":")
    if len(parts) == 3:
        domain = parts[0]
        # Truncate or pad domain to 4 chars
        if len(domain) > 4:
            domain = domain[-4:]  # Take last 4 (nvidia-smi uses 8-char 00000000)
        elif len(domain) < 4:
            domain = domain.zfill(4)
        return f"{domain}:{parts[1]}:{parts[2]}"
    return bdf


@dataclass
class GPUInfo:
    """Information about a single GPU, gathered from the rank that owns it."""

    global_rank: int
    local_rank: int  # rank index within the node (0..ranks_per_node-1)
    hostname: str
    gpu_id: int  # CUDA/HIP device index (local to process visibility)
    pci_bus_id: str  # e.g. "0000:41:00.0" — physical PCI address
    device_name: str  # e.g. "NVIDIA A100-SXM4-80GB" or "AMD Instinct MI300X"
    total_memory_mb: int
    numa_node: int  # NUMA node affinity (-1 if unknown)
    vendor: str  # "nvidia" or "amd"
    uuid: str  # GPU UUID
    fabric_info: FabricInfo = field(default_factory=FabricInfo)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "fabric_info"}
        d["fabric_info"] = self.fabric_info.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GPUInfo":
        # Don't mutate the input dict — use .get() and filter instead
        fabric_data = d.get("fabric_info", {})
        filtered = {k: v for k, v in d.items() if k != "fabric_info"}
        info = cls(**filtered)
        info.fabric_info = FabricInfo.from_dict(fabric_data)
        return info


@dataclass
class NodeInfo:
    """Aggregated information about a single physical host (node)."""

    hostname: str
    ranks: List[int] = field(default_factory=list)
    gpu_ids: List[int] = field(default_factory=list)  # gpu_id per rank (may have dups)
    unique_gpu_ids: List[int] = field(default_factory=list)  # deduplicated, sorted
    unique_pci_ids: List[str] = field(default_factory=list)  # deduplicated physical PCI bus IDs
    num_gpus: int = 0  # count of unique physical GPUs (by PCI bus ID), NOT ranks
    num_ranks: int = 0  # count of ranks on this node
    has_infiniband: bool = False
    ib_devices: List[str] = field(default_factory=list)
    # Intra-node link matrix: link_types[gpu_i][gpu_j] indexed by gpu_id (device index)
    link_types: Optional[List[List[int]]] = None
    # P2P access matrix: p2p_access[gpu_i][gpu_j] indexed by gpu_id (device index)
    p2p_access: Optional[List[List[bool]]] = None
    # Fabric domain keys for GPUs on this node
    fabric_domain_key: str = ""  # primary (for backward compat)
    fabric_domain_keys: List[str] = field(default_factory=list)  # all unique

    def get_link_type(self, gpu_id_a: int, gpu_id_b: int) -> IntraNodeLinkType:
        """
        Look up the link type between two GPUs by their device index (gpu_id).

        This is the safe accessor that avoids the oversubscription IndexError
        where local_rank > len(link_types matrix).
        """
        if self.link_types is None:
            return IntraNodeLinkType.UNKNOWN
        if gpu_id_a == gpu_id_b:
            return IntraNodeLinkType.SELF
        if gpu_id_a >= len(self.link_types) or gpu_id_b >= len(self.link_types[0]):
            return IntraNodeLinkType.UNKNOWN
        return IntraNodeLinkType(self.link_types[gpu_id_a][gpu_id_b])

    def can_p2p_access(self, gpu_id_a: int, gpu_id_b: int) -> bool:
        """
        Look up P2P accessibility between two GPUs by their device index (gpu_id).
        """
        if self.p2p_access is None:
            return gpu_id_a == gpu_id_b
        if gpu_id_a == gpu_id_b:
            return True
        if gpu_id_a >= len(self.p2p_access) or gpu_id_b >= len(self.p2p_access[0]):
            return False
        return self.p2p_access[gpu_id_a][gpu_id_b]


@dataclass
class TopologyMap:
    """
    Complete cluster topology, built from all-gathered GPU information.

    This is the primary output of topology discovery and is used by the
    hierarchical memory manager to decide communication strategies.
    """

    world_size: int
    num_nodes: int
    gpu_info: Dict[int, GPUInfo]  # rank -> GPUInfo
    nodes: Dict[str, NodeInfo]  # hostname -> NodeInfo
    fabric_domains: Dict[str, List[str]]  # domain_key -> [hostname, ...]
    # Precomputed peer groups (lazily populated)
    _node_peers: Dict[int, Set[int]] = field(default_factory=dict)
    _fabric_domain_peers: Dict[int, Set[int]] = field(default_factory=dict)

    def get_interconnect_level(self, rank_a: int, rank_b: int) -> InterconnectLevel:
        """
        Determine the interconnect tier between two ranks.

        Decision tree:
            1. Same hostname -> INTRA_NODE (IPC handles)
            2. Same fabric domain_key -> INTRA_RACK_FABRIC (NVLink/xGMI fabric)
            3. Otherwise -> INTER_NODE_RDMA (InfiniBand)
        """
        if rank_a == rank_b:
            return InterconnectLevel.INTRA_NODE

        info_a = self.gpu_info[rank_a]
        info_b = self.gpu_info[rank_b]

        # Same hostname -> same physical node
        if info_a.hostname == info_b.hostname:
            return InterconnectLevel.INTRA_NODE

        # Same fabric domain -> intra-rack fabric (NVLink domain / xGMI hive)
        key_a = info_a.fabric_info.domain_key
        key_b = info_b.fabric_info.domain_key
        if key_a and key_a == key_b:
            return InterconnectLevel.INTRA_RACK_FABRIC

        # Everything else -> RDMA
        return InterconnectLevel.INTER_NODE_RDMA

    def get_node_peers(self, rank: int) -> Set[int]:
        """Return all ranks on the same node as `rank` (excluding self)."""
        if rank not in self._node_peers:
            hostname = self.gpu_info[rank].hostname
            self._node_peers[rank] = {r for r, info in self.gpu_info.items() if info.hostname == hostname and r != rank}
        return self._node_peers[rank]

    def get_fabric_domain_peers(self, rank: int) -> Set[int]:
        """Return all ranks in the same fabric domain (excluding self)."""
        if rank not in self._fabric_domain_peers:
            domain_key = self.gpu_info[rank].fabric_info.domain_key
            if not domain_key:
                self._fabric_domain_peers[rank] = set()
            else:
                self._fabric_domain_peers[rank] = {
                    r for r, info in self.gpu_info.items() if info.fabric_info.domain_key == domain_key and r != rank
                }
        return self._fabric_domain_peers[rank]

    def get_rdma_peers(self, rank: int) -> Set[int]:
        """Return all ranks reachable only via RDMA."""
        all_ranks = set(self.gpu_info.keys())
        node_peers = self.get_node_peers(rank)
        fabric_peers = self.get_fabric_domain_peers(rank)
        return all_ranks - node_peers - fabric_peers - {rank}

    def get_ranks_for_node(self, hostname: str) -> List[int]:
        """Return sorted list of ranks on a given node."""
        if hostname in self.nodes:
            return sorted(self.nodes[hostname].ranks)
        return []

    def get_ranks_for_fabric_domain(self, domain_key: str) -> List[int]:
        """Return sorted list of all ranks in a fabric domain."""
        if domain_key not in self.fabric_domains:
            return []
        ranks = []
        for hostname in self.fabric_domains[domain_key]:
            ranks.extend(self.get_ranks_for_node(hostname))
        return sorted(ranks)

    def summary(self) -> str:
        """Human-readable summary of the topology."""
        lines = [
            "=== Iris Cluster Topology ===",
            f"World size: {self.world_size}  |  Nodes: {self.num_nodes}  |  Fabric domains: {len(self.fabric_domains)}",
            "",
        ]

        for hostname, node in sorted(self.nodes.items()):
            ib_str = f"IB: {', '.join(node.ib_devices)}" if node.has_infiniband else "IB: none"
            fabric_str = f"fabric: {node.fabric_domain_key}" if node.fabric_domain_key else ""
            oversubscribed = ""
            if node.num_ranks > node.num_gpus:
                oversubscribed = f" [oversubscribed: {node.num_ranks} ranks on {node.num_gpus} GPUs]"
            lines.append(
                f"  Node '{hostname}': {node.num_gpus} GPUs, ranks {node.ranks}, {ib_str} {fabric_str}{oversubscribed}"
            )
            for rank in sorted(node.ranks):
                info = self.gpu_info[rank]
                lines.append(
                    f"    rank {rank}: GPU{info.gpu_id} "
                    f"({info.device_name}, {info.total_memory_mb}MB) "
                    f"PCI={info.pci_bus_id} NUMA={info.numa_node}"
                )

        if self.fabric_domains:
            lines.append("")
            lines.append("Fabric Domains:")
            for domain_key, hostnames in sorted(self.fabric_domains.items()):
                total_gpus = sum(self.nodes[h].num_gpus for h in hostnames if h in self.nodes)
                lines.append(f"  {domain_key}: {len(hostnames)} nodes, {total_gpus} GPUs, hosts={hostnames}")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _detect_vendor() -> str:
    """Detect whether we're running on NVIDIA or AMD GPUs."""
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        return "amd"
    if torch.cuda.is_available():
        return "nvidia"
    return "unknown"


def _get_total_memory_mb(gpu_id: int) -> int:
    """
    Get total GPU memory in MB, compatible across PyTorch versions.

    Handles both `total_memory` (newer PyTorch) and `total_mem` (older)
    attribute names on device properties.
    """
    props = torch.cuda.get_device_properties(gpu_id)
    total_bytes = getattr(props, "total_memory", None)
    if total_bytes is None:
        total_bytes = getattr(props, "total_mem", 0)
    return total_bytes // (1024 * 1024)


def _get_pci_bus_id(device_idx: int, vendor: str) -> str:
    """
    Get the PCI bus ID for a GPU device.

    Uses _logical_to_physical_gpu_index() to translate the PyTorch
    logical device index to the physical index expected by NVML/AMDSMI,
    then queries NVML/AMDSMI by that physical index to obtain and
    normalize the busId/BDF string.
    """
    if device_idx < 0:
        logger.debug("Invalid device index: %d", device_idx)
        return "unknown"

    physical_idx = _logical_to_physical_gpu_index(device_idx, vendor)

    if vendor == "nvidia":
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(physical_idx)
                pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
                bus_id = pci_info.busId
                if isinstance(bus_id, bytes):
                    bus_id = bus_id.decode("utf-8")
                return _normalize_pci_bus_id(bus_id)
            finally:
                pynvml.nvmlShutdown()
        except ImportError:
            logger.debug("pynvml not available")
        except Exception as e:
            logger.debug(
                "NVML query failed for device %d (physical %d): %s",
                device_idx,
                physical_idx,
                e,
            )

    elif vendor == "amd":
        try:
            import amdsmi

            amdsmi.amdsmi_init()
            try:
                handles = amdsmi.amdsmi_get_processor_handles()
                if 0 <= physical_idx < len(handles):
                    bus_id = amdsmi.amdsmi_get_gpu_device_bdf(handles[physical_idx])
                    if bus_id:
                        return _normalize_pci_bus_id(str(bus_id))
                else:
                    logger.debug(
                        "physical_idx %d out of range (%d GPUs)",
                        physical_idx,
                        len(handles),
                    )
            finally:
                amdsmi.amdsmi_shut_down()
        except ImportError:
            logger.debug("amdsmi not available")
        except Exception as e:
            logger.debug(
                "amdsmi query failed for device %d (physical %d): %s",
                device_idx,
                physical_idx,
                e,
            )

    else:
        logger.debug("Unknown vendor: %s", vendor)

    return "unknown"


def _get_gpu_uuid(device_idx: int, vendor: str, pci_bus_id: str = "") -> str:
    """
    Get the unique UUID for a GPU device.

    When *pci_bus_id* is available, resolves the vendor handle via PCI
    address (immune to visibility env var issues).  Otherwise falls back
    to physical-index-based lookup via _logical_to_physical_gpu_index().
    """
    if device_idx < 0:
        logger.debug("Invalid device index: %d", device_idx)
        return f"gpu-{socket.gethostname()}-{device_idx}"

    physical_idx = _logical_to_physical_gpu_index(device_idx, vendor)

    if vendor == "nvidia":
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                # Prefer PCI-based resolution — immune to CUDA_VISIBLE_DEVICES
                handle = None
                if pci_bus_id and pci_bus_id != "unknown":
                    handle = _get_nvml_handle_by_pci(pci_bus_id)
                if handle is None:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_idx)

                uuid = pynvml.nvmlDeviceGetUUID(handle)
                if isinstance(uuid, bytes):
                    uuid = uuid.decode("utf-8")
                return uuid
            finally:
                pynvml.nvmlShutdown()
        except ImportError:
            logger.debug("pynvml not available")
        except Exception as e:
            logger.debug(
                "NVML UUID query failed for device %d (physical %d): %s",
                device_idx,
                physical_idx,
                e,
            )

    elif vendor == "amd":
        try:
            import amdsmi

            amdsmi.amdsmi_init()
            try:
                # Prefer PCI-based resolution
                handle = None
                if pci_bus_id and pci_bus_id != "unknown":
                    handle = _get_amdsmi_handle_by_pci(pci_bus_id)
                if handle is None:
                    handles = amdsmi.amdsmi_get_processor_handles()
                    if 0 <= physical_idx < len(handles):
                        handle = handles[physical_idx]

                if handle is not None:
                    uuid = amdsmi.amdsmi_get_gpu_device_uuid(handle)
                    if uuid:
                        return str(uuid)
                else:
                    logger.debug(
                        "physical_idx %d out of range or PCI resolve failed",
                        physical_idx,
                    )
            finally:
                amdsmi.amdsmi_shut_down()
        except ImportError:
            logger.debug("amdsmi not available")
        except Exception as e:
            logger.debug(
                "amdsmi UUID query failed for device %d (physical %d): %s",
                device_idx,
                physical_idx,
                e,
            )

    else:
        logger.debug("Unknown vendor: %s", vendor)

    return f"gpu-{socket.gethostname()}-{device_idx}"


def _get_numa_node(pci_bus_id: str) -> int:
    """
    Detect the NUMA node affinity for a GPU via sysfs.
    Uses the PCI bus ID to read the NUMA node from
    the kernel's PCI sysfs interface.
    Returns -1 if unknown.
    """
    if not pci_bus_id or pci_bus_id == "unknown":
        return -1

    pci_addr = pci_bus_id.lower()
    # sysfs always uses full domain prefix (0000:xx:xx.x)
    if len(pci_addr.split(":")) == 2:
        pci_addr = f"0000:{pci_addr}"

    numa_path = f"/sys/bus/pci/devices/{pci_addr}/numa_node"
    try:
        with open(numa_path) as f:
            node = int(f.read().strip())
            return node
    except (FileNotFoundError, ValueError, OSError) as e:
        logger.debug("Failed to read NUMA node from %s: %s", numa_path, e)
        return -1


def _detect_infiniband() -> Tuple[bool, List[str]]:
    """Detect InfiniBand/RDMA devices on this node via the kernel sysfs interface."""
    ib_class_path = "/sys/class/infiniband"
    try:
        if os.path.isdir(ib_class_path):
            devices = os.listdir(ib_class_path)
            if devices:
                return True, devices
    except OSError as e:
        logger.debug("Failed to read %s: %s", ib_class_path, e)

    return False, []


def _detect_intra_node_topology(
    local_pci_bus_ids: List[str],
    vendor: str,
) -> Tuple[Optional[List[List[int]]], Optional[List[List[bool]]]]:
    """
    Detect intra-node GPU-to-GPU topology (link types and P2P accessibility).

    Args:
        local_pci_bus_ids: PCI bus IDs of the GPUs visible to this process,
            indexed by local device index (matching torch.cuda device ordering).
            Used to correctly map physical CLI tool output to local indices.
        vendor: "nvidia" or "amd".

    Returns:
        (link_types, p2p_access) — both indexed by local device index (gpu_id).
        link_types may be None if CLI tools are unavailable.
    """
    num_gpus = len(local_pci_bus_ids)
    link_types = None
    p2p_access = None

    # P2P access detection uses PyTorch device indices, which already respect
    # visibility environment variables (e.g., CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES)
    # and any logical device remapping applied by the runtime.
    try:
        p2p_access = []
        for i in range(num_gpus):
            row = []
            for j in range(num_gpus):
                if i == j:
                    row.append(True)
                else:
                    row.append(torch.cuda.can_device_access_peer(i, j))
            p2p_access.append(row)
    except Exception as e:
        logger.debug(f"P2P access detection failed: {e}")
        p2p_access = None

    # Link type detection parses CLI tools that report physical topology.
    # We must map physical GPU indices to our local device indices via PCI bus IDs.
    if vendor == "nvidia":
        link_types = _parse_nvidia_topo(local_pci_bus_ids)
    elif vendor == "amd":
        link_types = _parse_amd_topo(local_pci_bus_ids)

    return link_types, p2p_access


# Map NVML topology levels to IntraNodeLinkType
_NVML_TOPO_TO_LINK = {
    0: IntraNodeLinkType.SELF,  # NVML_TOPOLOGY_INTERNAL
    10: IntraNodeLinkType.PCIE_SWITCH,  # NVML_TOPOLOGY_SINGLE
    20: IntraNodeLinkType.PCIE_SWITCH,  # NVML_TOPOLOGY_MULTIPLE
    30: IntraNodeLinkType.PCIE_HOST_BRIDGE,  # NVML_TOPOLOGY_HOSTBRIDGE
    40: IntraNodeLinkType.PCIE_NUMA,  # NVML_TOPOLOGY_NODE
    50: IntraNodeLinkType.PCIE_SYSTEM,  # NVML_TOPOLOGY_SYSTEM
}

# Max NVLink connections per GPU (18 for B200)
_MAX_NVLINK_LINKS = 18


def _nvml_check_nvlink(handle_src, handle_dst, pynvml) -> bool:
    """Check if any active NVLink connects two GPU handles."""
    try:
        pci_dst = pynvml.nvmlDeviceGetPciInfo(handle_dst)
        target_bus = pci_dst.busId
        if isinstance(target_bus, bytes):
            target_bus = target_bus.decode("utf-8")
        target_bus = target_bus.lower()
    except Exception:
        return False

    for link in range(_MAX_NVLINK_LINKS):
        try:
            state = pynvml.nvmlDeviceGetNvLinkState(handle_src, link)
            if not state:
                continue
            remote_pci = pynvml.nvmlDeviceGetNvLinkRemotePciInfo(handle_src, link)
            remote_bus = remote_pci.busId
            if isinstance(remote_bus, bytes):
                remote_bus = remote_bus.decode("utf-8")
            if remote_bus.lower() == target_bus:
                return True
        except Exception:
            break  # Link index doesn't exist; indices are contiguous from 0

    return False


def _parse_nvidia_topo(local_pci_bus_ids: List[str]) -> Optional[List[List[int]]]:
    """
    Build GPU-GPU link type matrix using pynvml topology queries.

    Args:
        local_pci_bus_ids: PCI bus IDs for each local PyTorch device index.

    Returns:
        Link type matrix indexed by local device index, or None on failure.
    """
    num_local = len(local_pci_bus_ids)

    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            # Resolve local PCI bus IDs directly to NVML handles
            handles = []
            for pci in local_pci_bus_ids:
                norm = _normalize_pci_bus_id(pci)
                if len(norm.split(":")) == 2:
                    norm = f"0000:{norm}"
                try:
                    handle = pynvml.nvmlDeviceGetHandleByPciBusId(norm.encode())
                    handles.append(handle)
                except pynvml.NVMLError as e:
                    logger.warning("Could not get NVML handle for PCI %s: %s", pci, e)
                    return None

            # Build the matrix
            matrix = [[IntraNodeLinkType.UNKNOWN] * num_local for _ in range(num_local)]
            for i in range(num_local):
                for j in range(num_local):
                    if i == j:
                        matrix[i][j] = IntraNodeLinkType.SELF
                        continue

                    # NVLink takes priority over PCIe topology level
                    if _nvml_check_nvlink(handles[i], handles[j], pynvml):
                        matrix[i][j] = IntraNodeLinkType.NVLINK
                        continue

                    try:
                        level = pynvml.nvmlDeviceGetTopologyCommonAncestor(handles[i], handles[j])
                        matrix[i][j] = _NVML_TOPO_TO_LINK.get(level, IntraNodeLinkType.UNKNOWN)
                    except pynvml.NVMLError:
                        matrix[i][j] = IntraNodeLinkType.UNKNOWN

            return matrix
        finally:
            pynvml.nvmlShutdown()

    except ImportError:
        logger.debug("pynvml not available for topology query")
        return None
    except Exception as e:
        logger.debug("NVML topology query failed: %s", e)
        return None


def _parse_amd_topo(local_pci_bus_ids: List[str]) -> Optional[List[List[int]]]:
    """
    Build GPU-GPU link type matrix using amdsmi topology queries.

    Args:
        local_pci_bus_ids: PCI bus IDs for each local PyTorch device index.

    Returns:
        Link type matrix indexed by local device index, or None on failure.
    """
    num_local = len(local_pci_bus_ids)

    try:
        import amdsmi

        amdsmi.amdsmi_init()
        try:
            all_handles = amdsmi.amdsmi_get_processor_handles()

            # Build BDF -> handle map for all physical GPUs
            bdf_to_handle = {}
            for handle in all_handles:
                try:
                    bdf = amdsmi.amdsmi_get_gpu_device_bdf(handle)
                    if bdf:
                        bdf_to_handle[_normalize_pci_bus_id(str(bdf))] = handle
                except Exception:
                    continue

            # Resolve local PCI bus IDs to amdsmi handles
            handles = []
            for pci in local_pci_bus_ids:
                norm = _normalize_pci_bus_id(pci)
                handle = bdf_to_handle.get(norm)
                if handle is None:
                    logger.warning(
                        "Could not find amdsmi handle for PCI %s. Known BDFs: %s",
                        pci,
                        list(bdf_to_handle.keys()),
                    )
                    return None
                handles.append(handle)

            # Pre-compute XGMI neighbor sets for each local GPU.
            # amdsmi_get_link_topology_nearest returns all GPUs reachable
            # via a given link type from a source GPU.
            xgmi_neighbors: List[Set[str]] = []
            for handle in handles:
                neighbors: Set[str] = set()
                try:
                    result = amdsmi.amdsmi_get_link_topology_nearest(
                        handle, amdsmi.AmdSmiLinkType.AMDSMI_LINK_TYPE_XGMI
                    )
                    for peer in result.get("processor_list", []):
                        try:
                            peer_bdf = amdsmi.amdsmi_get_gpu_device_bdf(peer)
                            if peer_bdf:
                                neighbors.add(_normalize_pci_bus_id(str(peer_bdf)))
                        except Exception:
                            continue
                except Exception:
                    pass  # No XGMI on this GPU (PCIe-only system)
                xgmi_neighbors.append(neighbors)

            # Normalized BDFs for each local device index
            local_bdfs = [_normalize_pci_bus_id(pci) for pci in local_pci_bus_ids]

            # Build the matrix
            matrix = [[IntraNodeLinkType.UNKNOWN] * num_local for _ in range(num_local)]
            for i in range(num_local):
                for j in range(num_local):
                    if i == j:
                        matrix[i][j] = IntraNodeLinkType.SELF
                        continue

                    # Check if GPU j's BDF is in GPU i's XGMI neighbor set
                    if local_bdfs[j] in xgmi_neighbors[i]:
                        matrix[i][j] = IntraNodeLinkType.NVLINK  # XGMI maps to NVLINK enum
                    else:
                        matrix[i][j] = IntraNodeLinkType.PCIE_SWITCH

            return matrix
        finally:
            amdsmi.amdsmi_shut_down()

    except ImportError:
        logger.debug("amdsmi not available for topology query")
        return None
    except Exception as e:
        logger.debug("amdsmi topology query failed: %s", e)
        return None


def _all_gather_strings(local_string: str, world_size: int) -> List[str]:
    """
    All-gather a string from each rank using PyTorch distributed.

    Compatible with Iris's use of torch.distributed (NCCL/RCCL backend).
    """
    local_bytes = local_string.encode("utf-8")
    local_len = len(local_bytes)

    # All-gather lengths
    len_tensor = torch.tensor([local_len], dtype=torch.long, device="cuda")
    len_list = [torch.zeros(1, dtype=torch.long, device="cuda") for _ in range(world_size)]
    dist.all_gather(len_list, len_tensor)
    max_len = max(t.item() for t in len_list)

    if max_len == 0:
        return [""] * world_size

    # All-gather padded byte tensors
    # Use frombuffer to avoid iterating every byte through Python space.
    # bytearray is required because frombuffer needs a writable buffer,
    # and copy=True ensures NCCL gets an owned contiguous CUDA tensor.
    padded = bytearray(local_bytes) + bytearray(max_len - local_len)
    local_tensor = torch.frombuffer(padded, dtype=torch.uint8).to("cuda", copy=True)
    gathered = [torch.zeros(max_len, dtype=torch.uint8, device="cuda") for _ in range(world_size)]
    dist.all_gather(gathered, local_tensor)

    results = []
    for t, length_t in zip(gathered, len_list):
        length = int(length_t.item())
        results.append(bytes(t[:length].cpu().tolist()).decode("utf-8"))
    return results


class TopologyDiscovery:
    """
    Multi-node multi-GPU topology discovery for Iris.

    Integrates with Iris's existing PyTorch distributed setup and produces
    a TopologyMap that classifies every GPU pair into one of three tiers
    based on hostname (intra-node), fabric info (intra-rack), or neither (RDMA).

    The fabric domain detection uses:
        AMD:    amdsmi_get_gpu_fabric_info -> (ppod_id, vpod_id)
        NVIDIA: nvmlDeviceGetGpuFabricInfoV -> (clusterUuid, cliqueId)
    """

    def __init__(self, iris_ctx=None):
        self._iris_ctx = iris_ctx
        self._topology: Optional[TopologyMap] = None

        if iris_ctx is not None:
            self.rank = iris_ctx.cur_rank
            self.world_size = iris_ctx.num_ranks
            self.gpu_id = iris_ctx.gpu_id
        else:
            num_gpus = torch.cuda.device_count()
            if num_gpus <= 0:
                raise RuntimeError("TopologyDiscovery requires at least one GPU")

            # Use LOCAL_RANK (set by torchrun/SLURM) for per-node GPU assignment.
            # This is more robust than global_rank % num_gpus, which breaks when
            # ranks aren't distributed in a way that aligns with device_count
            # (e.g., 2 nodes with 8 GPUs each but only 4 ranks per node).
            # The % num_gpus clamp handles isolation (LOCAL_RANK=3, device_count=1).
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.gpu_id = local_rank % num_gpus
            # MUST set device BEFORE init_process_group — NCCL needs a CUDA
            # device assigned to this process, otherwise all ranks fight over
            # GPU 0 and init either fails or produces world_size=1.
            torch.cuda.set_device(self.gpu_id)
            if dist.is_initialized():
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
            else:
                raise RuntimeError("TopologyDiscovery requires an initialized distributed process group.")

    @property
    def topology(self) -> Optional[TopologyMap]:
        """The discovered topology, or None if discover() hasn't been called."""
        return self._topology

    def discover(self) -> TopologyMap:
        """
        Perform topology discovery across the cluster.

        This is a collective operation — all ranks must call it.

        Steps:
            1. Each rank probes its local GPU (device name, PCI, NUMA, UUID)
            2. Each rank queries fabric info (ppod/vpod on AMD, clusterUuid/cliqueId on NVIDIA)
            3. Each rank probes node-level info (IB devices, intra-node topology)
            4. All-gather GPU info + node info across all ranks
            5. Build the global TopologyMap with fabric domain grouping
        """
        vendor = _detect_vendor()
        hostname = socket.gethostname()

        logger.debug(
            f"[Rank {self.rank}] Starting topology discovery on {hostname}, GPU {self.gpu_id}, vendor={vendor}"
        )

        # Probe local GPU info
        device_name = torch.cuda.get_device_name(self.gpu_id)
        total_memory_mb = _get_total_memory_mb(self.gpu_id)
        pci_bus_id = _get_pci_bus_id(self.gpu_id, vendor)
        # Pass PCI bus ID to UUID query for PCI-based handle resolution
        gpu_uuid = _get_gpu_uuid(self.gpu_id, vendor, pci_bus_id=pci_bus_id)
        # Pass PCI bus ID
        numa_node = _get_numa_node(pci_bus_id)

        # Query fabric info — pass PCI bus ID for PCI-based handle resolution
        fabric_info = get_gpu_fabric_info(self.gpu_id, vendor, pci_bus_id=pci_bus_id)
        logger.debug(
            f"[Rank {self.rank}] Fabric info: cluster_uuid={fabric_info.cluster_uuid}, "
            f"clique_id={fabric_info.clique_id}, domain_key={fabric_info.domain_key}"
        )

        local_gpu_info = GPUInfo(
            global_rank=self.rank,
            local_rank=self.gpu_id,
            hostname=hostname,
            gpu_id=self.gpu_id,
            pci_bus_id=pci_bus_id,
            device_name=device_name,
            total_memory_mb=total_memory_mb,
            numa_node=numa_node,
            vendor=vendor,
            uuid=gpu_uuid,
            fabric_info=fabric_info,
        )

        # Probe node-level info
        # Gather PCI bus IDs for ALL visible GPUs on this node (not just this rank's).
        # This is needed for correct CLI tool output remapping.
        num_local_gpus = torch.cuda.device_count()
        if num_local_gpus <= 1 and self.world_size > 1:
            logger.warning(
                f"[Rank {self.rank}] torch.cuda.device_count() = {num_local_gpus}. "
                f"CUDA_VISIBLE_DEVICES may be restricting GPU visibility. "
                f"Intra-node topology detection will be limited."
            )

        local_pci_bus_ids = []
        for dev_idx in range(num_local_gpus):
            local_pci_bus_ids.append(_get_pci_bus_id(dev_idx, vendor))

        has_ib, ib_devices = _detect_infiniband()
        link_types, p2p_access = _detect_intra_node_topology(local_pci_bus_ids, vendor)

        # All-gather
        local_gpu_json = json.dumps(local_gpu_info.to_dict())
        all_gpu_jsons = _all_gather_strings(local_gpu_json, self.world_size)

        node_info_json = json.dumps(
            {
                "link_types": link_types,
                "p2p_access": p2p_access,
                "has_ib": has_ib,
                "ib_devices": ib_devices,
                "num_visible_gpus": num_local_gpus,
            }
        )
        all_node_jsons = _all_gather_strings(node_info_json, self.world_size)

        # Build global topology
        gpu_info_map: Dict[int, GPUInfo] = {}
        for gpu_json in all_gpu_jsons:
            info = GPUInfo.from_dict(json.loads(gpu_json))
            gpu_info_map[info.global_rank] = info

        all_node_infos = [json.loads(s) for s in all_node_jsons]

        # Group ranks by hostname
        hostname_to_ranks: Dict[str, List[int]] = {}
        for rank, info in gpu_info_map.items():
            hostname_to_ranks.setdefault(info.hostname, []).append(rank)

        # local_rank ordering: assign sequential index within each node
        for hostname, ranks in hostname_to_ranks.items():
            for local_idx, rank in enumerate(sorted(ranks)):
                gpu_info_map[rank].local_rank = local_idx

        # Build NodeInfo
        nodes: Dict[str, NodeInfo] = {}
        for hostname, ranks in hostname_to_ranks.items():
            sorted_ranks = sorted(ranks)
            representative = sorted_ranks[0]
            nd = all_node_infos[representative]

            # Collect per-rank gpu_ids and deduplicate physical GPUs by PCI bus ID.
            gpu_ids_per_rank = [gpu_info_map[r].gpu_id for r in sorted_ranks]
            pci_ids_per_rank = [gpu_info_map[r].pci_bus_id for r in sorted_ranks]
            unique_pci_ids = sorted(set(p for p in pci_ids_per_rank if p != "unknown"))
            unique_gpu_ids = sorted(set(gpu_ids_per_rank))
            num_physical_gpus = len(unique_pci_ids) if unique_pci_ids else len(unique_gpu_ids)

            # Collect all unique fabric domain keys and warn if mixed
            domain_keys: Set[str] = set()
            for r in sorted_ranks:
                dk = gpu_info_map[r].fabric_info.domain_key
                if dk:
                    domain_keys.add(dk)

            if len(domain_keys) > 1:
                logger.warning(
                    f"Node '{hostname}' has GPUs in multiple fabric domains: "
                    f"{domain_keys}. This may indicate a misconfiguration."
                )

            sorted_domain_keys = sorted(domain_keys)
            node_domain_key = sorted_domain_keys[0] if sorted_domain_keys else ""

            # Log oversubscription
            if len(sorted_ranks) > num_physical_gpus:
                logger.info(
                    f"Node '{hostname}': {len(sorted_ranks)} ranks oversubscribed "
                    f"on {num_physical_gpus} physical GPUs "
                    f"(pci_ids={unique_pci_ids or unique_gpu_ids})"
                )

            nodes[hostname] = NodeInfo(
                hostname=hostname,
                ranks=sorted_ranks,
                gpu_ids=gpu_ids_per_rank,
                unique_gpu_ids=unique_gpu_ids,
                unique_pci_ids=unique_pci_ids,
                num_gpus=num_physical_gpus,
                num_ranks=len(sorted_ranks),
                has_infiniband=nd["has_ib"],
                ib_devices=nd["ib_devices"],
                link_types=nd["link_types"],
                p2p_access=nd["p2p_access"],
                fabric_domain_key=node_domain_key,
                fabric_domain_keys=sorted_domain_keys,
            )

        # Build fabric domain map by registering each node under ALL its domains
        fabric_domains: Dict[str, List[str]] = {}
        for hostname, node in nodes.items():
            for dk in node.fabric_domain_keys:
                fabric_domains.setdefault(dk, []).append(hostname)

        self._topology = TopologyMap(
            world_size=self.world_size,
            num_nodes=len(nodes),
            gpu_info=gpu_info_map,
            nodes=nodes,
            fabric_domains=fabric_domains,
        )

        logger.debug(
            f"[Rank {self.rank}] Topology discovery complete: {len(nodes)} nodes, {len(fabric_domains)} fabric domains"
        )
        return self._topology

    def get_communication_groups(self) -> Dict[InterconnectLevel, List[List[int]]]:
        """
        Build communication groups for each interconnect level.
        """
        if self._topology is None:
            raise RuntimeError("Must call discover() before get_communication_groups()")

        topo = self._topology

        # Intra-node: one group per physical node (sorted by hostname for determinism)
        intra_node_groups = [
            sorted(node.ranks) for hostname, node in sorted(topo.nodes.items(), key=lambda item: item[0])
        ]

        # Fabric-level: group by fabric domain, plus standalone nodes
        if topo.fabric_domains:
            # Sort fabric domains for deterministic ordering of fabric groups
            fabric_domain_keys = sorted(topo.fabric_domains)
            fabric_groups = [sorted(topo.get_ranks_for_fabric_domain(dk)) for dk in fabric_domain_keys]

            # Include standalone nodes (no fabric domain) as their own groups
            # so they aren't orphaned from the fabric tier.
            nodes_in_fabric = set()
            for hostnames in topo.fabric_domains.values():
                nodes_in_fabric.update(hostnames)
            for hostname, node in sorted(topo.nodes.items(), key=lambda item: item[0]):
                if hostname not in nodes_in_fabric:
                    fabric_groups.append(sorted(node.ranks))
        else:
            # No fabric at all
            fabric_groups = []

        # RDMA: everyone
        rdma_groups = [sorted(topo.gpu_info.keys())]

        return {
            InterconnectLevel.INTRA_NODE: intra_node_groups,
            InterconnectLevel.INTRA_RACK_FABRIC: fabric_groups,
            InterconnectLevel.INTER_NODE_RDMA: rdma_groups,
        }

    def get_heap_distribution_plan(self) -> Dict[int, Dict[str, Any]]:
        """
        Generate a plan for distributing symmetric heap bases across the cluster.

        For each rank, classifies every peer into one of three tiers:
            - ipc_peers:    Same node -> use cudaIpcMemHandle / hipIpcMemHandle
            - fabric_peers: Same fabric domain, different node -> use fabric
                            memory handles (cuMemExportToShareableHandle on NVIDIA,
                            or equivalent on AMD)
            - rdma_peers:   Different fabric domain -> use RDMA
        """
        if self._topology is None:
            raise RuntimeError("Must call discover() before get_heap_distribution_plan()")

        topo = self._topology
        plan: Dict[int, Dict[str, Any]] = {}

        for rank, info in topo.gpu_info.items():
            ipc_peers = sorted(topo.get_node_peers(rank))
            fabric_peers = sorted(topo.get_fabric_domain_peers(rank) - topo.get_node_peers(rank))
            rdma_peers = sorted(topo.get_rdma_peers(rank))

            plan[rank] = {
                "ipc_peers": ipc_peers,
                "fabric_peers": fabric_peers,
                "rdma_peers": rdma_peers,
                "node": info.hostname,
                "local_rank": info.local_rank,
                "gpu_id": info.gpu_id,
                "fabric_domain": info.fabric_info.domain_key,
            }

        return plan
