# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for iris.topology — Multi-GPU topology discovery.

Some tests are pure unit tests (no GPU needed), others require a distributed
process group with real GPUs. The distributed tests are marked and will skip
gracefully if the environment isn't set up.
"""

import json
import socket

import pytest
import torch
import torch.distributed as dist

from iris.topology import (
    FabricInfo,
    GPUInfo,
    IntraNodeLinkType,
    InterconnectLevel,
    NodeInfo,
    TopologyDiscovery,
    TopologyMap,
    _all_gather_strings,
    _normalize_pci_bus_id,
    _logical_to_physical_gpu_index,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


# ---------------------------------------------------------------------------
# Unit tests — no GPU or distributed required
# ---------------------------------------------------------------------------


class TestFabricInfo:
    """Tests for FabricInfo data class."""

    def test_empty_fabric_info(self):
        fi = FabricInfo()
        assert fi.cluster_uuid == ""
        assert fi.clique_id == 0
        assert fi.is_valid is False
        assert fi.domain_key == ""

    def test_valid_fabric_info(self):
        fi = FabricInfo(cluster_uuid="00aabbccdd112233", clique_id=1)
        assert fi.is_valid is True
        assert fi.domain_key == "00aabbccdd112233:1"

    def test_domain_key_comparison(self):
        fi_a = FabricInfo(cluster_uuid="aabb", clique_id=0)
        fi_b = FabricInfo(cluster_uuid="aabb", clique_id=0)
        fi_c = FabricInfo(cluster_uuid="aabb", clique_id=1)
        fi_d = FabricInfo(cluster_uuid="ccdd", clique_id=0)
        assert fi_a.domain_key == fi_b.domain_key
        assert fi_a.domain_key != fi_c.domain_key  # different clique
        assert fi_a.domain_key != fi_d.domain_key  # different cluster

    def test_empty_domain_keys_do_not_match(self):
        """Empty domain keys are equal but falsy, so topology code won't match them."""
        fi_a = FabricInfo()
        fi_b = FabricInfo()
        assert fi_a.domain_key == fi_b.domain_key  # "" == "" is True
        assert not fi_a.domain_key

    def test_serialization_roundtrip(self):
        fi = FabricInfo(cluster_uuid="deadbeef", clique_id=42)
        d = fi.to_dict()
        fi2 = FabricInfo.from_dict(d)
        assert fi2.cluster_uuid == fi.cluster_uuid
        assert fi2.clique_id == fi.clique_id

    def test_from_dict_missing_keys(self):
        fi = FabricInfo.from_dict({})
        assert fi.cluster_uuid == ""
        assert fi.clique_id == 0


class TestGPUInfo:
    """Tests for GPUInfo data class."""

    def _make_gpu_info(
        self,
        rank=0,
        hostname="node-a",
        gpu_id=0,
        pci_bus_id="0000:41:00.0",
        fabric=None,
    ):
        return GPUInfo(
            global_rank=rank,
            local_rank=gpu_id,
            hostname=hostname,
            gpu_id=gpu_id,
            pci_bus_id=pci_bus_id,
            device_name="Test GPU",
            total_memory_mb=81920,
            numa_node=0,
            vendor="amd",
            uuid=f"gpu-{hostname}-{gpu_id}",
            fabric_info=fabric or FabricInfo(),
        )

    def test_serialization_roundtrip(self):
        fi = FabricInfo(cluster_uuid="aabb", clique_id=1)
        gpu = self._make_gpu_info(fabric=fi)
        d = gpu.to_dict()
        gpu2 = GPUInfo.from_dict(d)
        assert gpu2.global_rank == gpu.global_rank
        assert gpu2.hostname == gpu.hostname
        assert gpu2.pci_bus_id == gpu.pci_bus_id
        assert gpu2.fabric_info.cluster_uuid == "aabb"
        assert gpu2.fabric_info.clique_id == 1

    def test_from_dict_does_not_mutate_input(self):
        """Regression: old code used dict.pop() which mutated the input."""
        d = {
            "global_rank": 0,
            "local_rank": 0,
            "hostname": "h",
            "gpu_id": 0,
            "pci_bus_id": "x",
            "device_name": "x",
            "total_memory_mb": 0,
            "numa_node": 0,
            "vendor": "amd",
            "uuid": "x",
            "fabric_info": {"cluster_uuid": "abc", "clique_id": 1},
        }
        original_keys = set(d.keys())
        GPUInfo.from_dict(d)
        assert set(d.keys()) == original_keys
        assert "fabric_info" in d  # must not have been popped

    def test_from_dict_missing_fabric(self):
        d = {
            "global_rank": 0,
            "local_rank": 0,
            "hostname": "h",
            "gpu_id": 0,
            "pci_bus_id": "x",
            "device_name": "x",
            "total_memory_mb": 0,
            "numa_node": 0,
            "vendor": "amd",
            "uuid": "x",
        }
        gpu = GPUInfo.from_dict(d)
        assert gpu.fabric_info.is_valid is False


class TestNormalizePCIBusId:
    """Tests for PCI bus ID normalization."""

    def test_standard_format(self):
        assert _normalize_pci_bus_id("0000:41:00.0") == "0000:41:00.0"

    def test_uppercase(self):
        assert _normalize_pci_bus_id("0000:4A:00.0") == "0000:4a:00.0"

    def test_nvidia_8char_domain(self):
        """nvidia-smi sometimes uses 8-char domain like 00000000:41:00.0"""
        result = _normalize_pci_bus_id("00000000:41:00.0")
        assert result == "0000:41:00.0"

    def test_prefix_junk(self):
        result = _normalize_pci_bus_id("GPU 0000:41:00.0")
        assert result == "0000:41:00.0"

    def test_no_match(self):
        result = _normalize_pci_bus_id("garbage")
        assert result == "garbage"


class TestNodeInfo:
    """Tests for NodeInfo safe accessors."""

    def test_get_link_type_self(self):
        node = NodeInfo(
            hostname="h",
            link_types=[
                [IntraNodeLinkType.SELF, IntraNodeLinkType.NVLINK],
                [IntraNodeLinkType.NVLINK, IntraNodeLinkType.SELF],
            ],
        )
        assert node.get_link_type(0, 0) == IntraNodeLinkType.SELF
        assert node.get_link_type(0, 1) == IntraNodeLinkType.NVLINK

    def test_get_link_type_out_of_bounds(self):
        """Oversubscription safety: local_rank=3 on a 2-GPU node."""
        node = NodeInfo(
            hostname="h",
            link_types=[
                [IntraNodeLinkType.SELF, IntraNodeLinkType.NVLINK],
                [IntraNodeLinkType.NVLINK, IntraNodeLinkType.SELF],
            ],
        )
        # gpu_id 3 is out of bounds for a 2x2 matrix
        assert node.get_link_type(3, 0) == IntraNodeLinkType.UNKNOWN
        assert node.get_link_type(0, 3) == IntraNodeLinkType.UNKNOWN

    def test_get_link_type_no_matrix(self):
        node = NodeInfo(hostname="h", link_types=None)
        assert node.get_link_type(0, 1) == IntraNodeLinkType.UNKNOWN

    def test_p2p_access_out_of_bounds(self):
        node = NodeInfo(
            hostname="h",
            p2p_access=[[True, True], [True, True]],
        )
        assert node.can_p2p_access(0, 1) is True
        assert node.can_p2p_access(3, 0) is False  # out of bounds -> False

    def test_p2p_access_self_always_true(self):
        node = NodeInfo(hostname="h", p2p_access=None)
        assert node.can_p2p_access(0, 0) is True


class TestTopologyMap:
    """Tests for TopologyMap with synthetic data."""

    def _make_topology(self):
        """
        Build a synthetic 8-rank topology:
            node-a: ranks 0,1,2,3 (GPU0-3), fabric "aabb:0"
            node-b: ranks 4,5 (GPU0-1), fabric "aabb:0"
            node-c: ranks 6,7 (GPU0-1), no fabric
        """
        fabric_ab = FabricInfo(cluster_uuid="aabb", clique_id=0)
        gpus = {}
        for r in range(4):
            gpus[r] = GPUInfo(
                global_rank=r,
                local_rank=r,
                hostname="node-a",
                gpu_id=r,
                pci_bus_id=f"0000:4{r}:00.0",
                device_name="MI300X",
                total_memory_mb=81920,
                numa_node=0,
                vendor="amd",
                uuid=f"gpu-a-{r}",
                fabric_info=fabric_ab,
            )
        for r in range(4, 6):
            gpus[r] = GPUInfo(
                global_rank=r,
                local_rank=r - 4,
                hostname="node-b",
                gpu_id=r - 4,
                pci_bus_id=f"0000:8{r - 4}:00.0",
                device_name="MI300X",
                total_memory_mb=81920,
                numa_node=0,
                vendor="amd",
                uuid=f"gpu-b-{r - 4}",
                fabric_info=fabric_ab,
            )
        for r in range(6, 8):
            gpus[r] = GPUInfo(
                global_rank=r,
                local_rank=r - 6,
                hostname="node-c",
                gpu_id=r - 6,
                pci_bus_id=f"0000:c{r - 6}:00.0",
                device_name="A100",
                total_memory_mb=81920,
                numa_node=0,
                vendor="nvidia",
                uuid=f"gpu-c-{r - 6}",
                fabric_info=FabricInfo(),
            )

        nodes = {
            "node-a": NodeInfo(
                hostname="node-a",
                ranks=[0, 1, 2, 3],
                gpu_ids=[0, 1, 2, 3],
                unique_gpu_ids=[0, 1, 2, 3],
                unique_pci_ids=[
                    "0000:40:00.0",
                    "0000:41:00.0",
                    "0000:42:00.0",
                    "0000:43:00.0",
                ],
                num_gpus=4,
                num_ranks=4,
                fabric_domain_key="aabb:0",
                fabric_domain_keys=["aabb:0"],
            ),
            "node-b": NodeInfo(
                hostname="node-b",
                ranks=[4, 5],
                gpu_ids=[0, 1],
                unique_gpu_ids=[0, 1],
                unique_pci_ids=["0000:80:00.0", "0000:81:00.0"],
                num_gpus=2,
                num_ranks=2,
                fabric_domain_key="aabb:0",
                fabric_domain_keys=["aabb:0"],
            ),
            "node-c": NodeInfo(
                hostname="node-c",
                ranks=[6, 7],
                gpu_ids=[0, 1],
                unique_gpu_ids=[0, 1],
                unique_pci_ids=["0000:c0:00.0", "0000:c1:00.0"],
                num_gpus=2,
                num_ranks=2,
                fabric_domain_key="",
                fabric_domain_keys=[],
            ),
        }

        fabric_domains = {"aabb:0": ["node-a", "node-b"]}

        return TopologyMap(
            world_size=8,
            num_nodes=3,
            gpu_info=gpus,
            nodes=nodes,
            fabric_domains=fabric_domains,
        )

    # --- Interconnect level classification ---

    def test_same_rank_is_intra_node(self):
        topo = self._make_topology()
        assert topo.get_interconnect_level(0, 0) == InterconnectLevel.INTRA_NODE

    def test_same_node_is_intra_node(self):
        topo = self._make_topology()
        assert topo.get_interconnect_level(0, 3) == InterconnectLevel.INTRA_NODE
        assert topo.get_interconnect_level(1, 2) == InterconnectLevel.INTRA_NODE

    def test_same_fabric_different_node_is_fabric(self):
        topo = self._make_topology()
        # rank 0 (node-a) <-> rank 4 (node-b): both in fabric "aabb:0"
        assert topo.get_interconnect_level(0, 4) == InterconnectLevel.INTRA_RACK_FABRIC
        assert topo.get_interconnect_level(3, 5) == InterconnectLevel.INTRA_RACK_FABRIC

    def test_no_fabric_is_rdma(self):
        topo = self._make_topology()
        # rank 0 (node-a, fabric) <-> rank 6 (node-c, no fabric)
        assert topo.get_interconnect_level(0, 6) == InterconnectLevel.INTER_NODE_RDMA
        assert topo.get_interconnect_level(4, 7) == InterconnectLevel.INTER_NODE_RDMA

    # --- Peer groups ---

    def test_node_peers(self):
        topo = self._make_topology()
        assert topo.get_node_peers(0) == {1, 2, 3}
        assert topo.get_node_peers(4) == {5}
        assert topo.get_node_peers(6) == {7}

    def test_fabric_domain_peers(self):
        topo = self._make_topology()
        # rank 0: fabric peers = all of (node-a + node-b) minus self
        assert topo.get_fabric_domain_peers(0) == {1, 2, 3, 4, 5}
        # rank 4: fabric peers = all of (node-a + node-b) minus self
        assert topo.get_fabric_domain_peers(4) == {0, 1, 2, 3, 5}
        # rank 6: no fabric -> empty
        assert topo.get_fabric_domain_peers(6) == set()

    def test_rdma_peers(self):
        topo = self._make_topology()
        # rank 0: RDMA peers = everyone not in same node or fabric = {6, 7}
        assert topo.get_rdma_peers(0) == {6, 7}
        # rank 6: RDMA peers = everyone not on node-c = {0,1,2,3,4,5}
        assert topo.get_rdma_peers(6) == {0, 1, 2, 3, 4, 5}

    def test_peer_groups_partition_world(self):
        """Node peers + fabric-only peers + RDMA peers + self = world."""
        topo = self._make_topology()
        for rank in range(8):
            node = topo.get_node_peers(rank)
            fabric_only = topo.get_fabric_domain_peers(rank) - node
            rdma = topo.get_rdma_peers(rank)
            all_peers = node | fabric_only | rdma | {rank}
            assert all_peers == set(range(8)), (
                f"Rank {rank}: partition incomplete. Missing: {set(range(8)) - all_peers}"
            )
            # No overlaps
            assert not (node & rdma), f"Rank {rank}: node∩rdma overlap"
            assert not (fabric_only & rdma), f"Rank {rank}: fabric∩rdma overlap"

    # --- Communication groups ---

    def test_comm_groups_intra_node(self):
        topo = self._make_topology()
        td = TopologyDiscovery.__new__(TopologyDiscovery)
        td._topology = topo
        groups = td.get_communication_groups()
        intra = groups[InterconnectLevel.INTRA_NODE]
        assert sorted(intra, key=lambda g: g[0]) == [
            [0, 1, 2, 3],
            [4, 5],
            [6, 7],
        ]

    def test_comm_groups_fabric_includes_standalone(self):
        """
        When fabric domains exist, standalone nodes (no fabric) must still
        appear in the fabric tier so they aren't orphaned.
        """
        topo = self._make_topology()
        td = TopologyDiscovery.__new__(TopologyDiscovery)
        td._topology = topo
        groups = td.get_communication_groups()
        fabric = groups[InterconnectLevel.INTRA_RACK_FABRIC]
        # Should have 2 groups: fabric domain + standalone
        all_ranks_in_fabric = set()
        for g in fabric:
            all_ranks_in_fabric.update(g)
        assert all_ranks_in_fabric == set(range(8)), (
            f"Ranks missing from fabric groups: {set(range(8)) - all_ranks_in_fabric}"
        )

    def test_comm_groups_fabric_domain_group_content(self):
        """Fabric domain group should contain exactly the ranks in that domain."""
        topo = self._make_topology()
        td = TopologyDiscovery.__new__(TopologyDiscovery)
        td._topology = topo
        groups = td.get_communication_groups()
        fabric = groups[InterconnectLevel.INTRA_RACK_FABRIC]
        # First group should be the "aabb:0" domain (node-a + node-b)
        assert [0, 1, 2, 3, 4, 5] in fabric
        # Second group should be the standalone node-c
        assert [6, 7] in fabric

    def test_comm_groups_fabric_is_not_empty_when_domains_exist(self):
        """When fabric domains exist, fabric groups must be non-empty."""
        topo = self._make_topology()
        td = TopologyDiscovery.__new__(TopologyDiscovery)
        td._topology = topo
        groups = td.get_communication_groups()
        assert len(groups[InterconnectLevel.INTRA_RACK_FABRIC]) > 0

    def test_comm_groups_rdma_is_world(self):
        topo = self._make_topology()
        td = TopologyDiscovery.__new__(TopologyDiscovery)
        td._topology = topo
        groups = td.get_communication_groups()
        assert groups[InterconnectLevel.INTER_NODE_RDMA] == [list(range(8))]

    # --- Heap distribution plan ---

    def test_heap_plan_completeness(self):
        topo = self._make_topology()
        td = TopologyDiscovery.__new__(TopologyDiscovery)
        td._topology = topo
        plan = td.get_heap_distribution_plan()
        assert set(plan.keys()) == set(range(8))

    def test_heap_plan_rank4(self):
        topo = self._make_topology()
        td = TopologyDiscovery.__new__(TopologyDiscovery)
        td._topology = topo
        plan = td.get_heap_distribution_plan()
        p4 = plan[4]
        assert p4["ipc_peers"] == [5]  # same node
        assert p4["fabric_peers"] == [0, 1, 2, 3]  # same fabric, diff node
        assert p4["rdma_peers"] == [6, 7]  # no fabric

    def test_heap_plan_no_peer_overlap(self):
        topo = self._make_topology()
        td = TopologyDiscovery.__new__(TopologyDiscovery)
        td._topology = topo
        plan = td.get_heap_distribution_plan()
        for rank, p in plan.items():
            ipc = set(p["ipc_peers"])
            fabric = set(p["fabric_peers"])
            rdma = set(p["rdma_peers"])
            assert not (ipc & fabric), f"Rank {rank}: ipc∩fabric"
            assert not (ipc & rdma), f"Rank {rank}: ipc∩rdma"
            assert not (fabric & rdma), f"Rank {rank}: fabric∩rdma"
            assert rank not in (ipc | fabric | rdma), f"Rank {rank}: self in peers"

    # --- Topology summary ---

    def test_summary_contains_all_nodes(self):
        topo = self._make_topology()
        s = topo.summary()
        assert "node-a" in s
        assert "node-b" in s
        assert "node-c" in s
        assert "Fabric Domains" in s

    def test_ranks_for_fabric_domain(self):
        topo = self._make_topology()
        ranks = topo.get_ranks_for_fabric_domain("aabb:0")
        assert ranks == [0, 1, 2, 3, 4, 5]

    def test_ranks_for_nonexistent_domain(self):
        topo = self._make_topology()
        assert topo.get_ranks_for_fabric_domain("nonexistent") == []


class TestOversubscription:
    """
    Tests for the oversubscription scenario:
    4 ranks sharing 2 physical GPUs on the same node.
    """

    def _make_oversubscribed_topology(self):
        """
        node-x: ranks 0,1,2,3
            ranks 0,1 -> gpu_id=0, PCI=0000:41:00.0
            ranks 2,3 -> gpu_id=1, PCI=0000:42:00.0
        """
        gpus = {}
        for r in range(4):
            gid = r // 2
            gpus[r] = GPUInfo(
                global_rank=r,
                local_rank=r,
                hostname="node-x",
                gpu_id=gid,
                pci_bus_id=f"0000:4{gid + 1}:00.0",
                device_name="MI300X",
                total_memory_mb=81920,
                numa_node=0,
                vendor="amd",
                uuid=f"gpu-x-{gid}",
            )

        nodes = {
            "node-x": NodeInfo(
                hostname="node-x",
                ranks=[0, 1, 2, 3],
                gpu_ids=[0, 0, 1, 1],
                unique_gpu_ids=[0, 1],
                unique_pci_ids=["0000:41:00.0", "0000:42:00.0"],
                num_gpus=2,  # 2 physical GPUs, not 4 ranks
                num_ranks=4,
                link_types=[
                    [IntraNodeLinkType.SELF, IntraNodeLinkType.NVLINK],
                    [IntraNodeLinkType.NVLINK, IntraNodeLinkType.SELF],
                ],
                p2p_access=[[True, True], [True, True]],
            ),
        }

        return TopologyMap(
            world_size=4,
            num_nodes=1,
            gpu_info=gpus,
            nodes=nodes,
            fabric_domains={},
        )

    def test_num_gpus_is_physical_count(self):
        topo = self._make_oversubscribed_topology()
        assert topo.nodes["node-x"].num_gpus == 2
        assert topo.nodes["node-x"].num_ranks == 4

    def test_link_type_by_gpu_id(self):
        """Topology lookup uses gpu_id (device index), not local_rank."""
        node = self._make_oversubscribed_topology().nodes["node-x"]
        assert node.get_link_type(0, 1) == IntraNodeLinkType.NVLINK
        assert node.get_link_type(0, 0) == IntraNodeLinkType.SELF

    def test_p2p_by_gpu_id(self):
        node = self._make_oversubscribed_topology().nodes["node-x"]
        assert node.can_p2p_access(0, 1) is True

    def test_all_ranks_are_node_peers(self):
        topo = self._make_oversubscribed_topology()
        assert topo.get_node_peers(0) == {1, 2, 3}
        assert topo.get_node_peers(2) == {0, 1, 3}


class TestIsolationCollapse:
    """
    Tests for the GPU isolation (CUDA_VISIBLE_DEVICES per-process) scenario.

    When SLURM/K8s isolates GPUs, every rank reports gpu_id=0, but their
    PCI bus IDs differ. The node must NOT collapse to num_gpus=1.
    """

    def _make_isolated_topology(self):
        """
        node-y: ranks 0,1
            rank 0 -> gpu_id=0 (isolated), PCI=0000:c1:00.0
            rank 1 -> gpu_id=0 (isolated), PCI=0000:c2:00.0
        """
        gpus = {}
        for r in range(2):
            gpus[r] = GPUInfo(
                global_rank=r,
                local_rank=r,
                hostname="node-y",
                gpu_id=0,  # both report 0 due to isolation
                pci_bus_id=f"0000:c{r + 1}:00.0",  # but different physical GPUs
                device_name="A100",
                total_memory_mb=81920,
                numa_node=0,
                vendor="nvidia",
                uuid=f"gpu-y-{r}",
            )

        nodes = {
            "node-y": NodeInfo(
                hostname="node-y",
                ranks=[0, 1],
                gpu_ids=[0, 0],
                unique_gpu_ids=[0],  # gpu_id dedup gives 1
                unique_pci_ids=["0000:c1:00.0", "0000:c2:00.0"],
                num_gpus=2,  # PCI dedup correctly gives 2
                num_ranks=2,
            ),
        }

        return TopologyMap(
            world_size=2,
            num_nodes=1,
            gpu_info=gpus,
            nodes=nodes,
            fabric_domains={},
        )

    def test_num_gpus_not_collapsed(self):
        """
        Regression: with gpu_id dedup, num_gpus would be 1.
        With PCI dedup, it's correctly 2.
        """
        topo = self._make_isolated_topology()
        assert topo.nodes["node-y"].num_gpus == 2

    def test_both_ranks_are_node_peers(self):
        topo = self._make_isolated_topology()
        assert topo.get_node_peers(0) == {1}
        assert topo.get_node_peers(1) == {0}


class TestNoFabricCluster:
    """Tests for a cluster with NO fabric domains at all."""

    def _make_no_fabric_topology(self):
        gpus = {}
        for r in range(4):
            node = "node-a" if r < 2 else "node-b"
            gpus[r] = GPUInfo(
                global_rank=r,
                local_rank=r % 2,
                hostname=node,
                gpu_id=r % 2,
                pci_bus_id=f"0000:{r}0:00.0",
                device_name="T4",
                total_memory_mb=16384,
                numa_node=0,
                vendor="nvidia",
                uuid=f"gpu-{r}",
            )

        nodes = {
            "node-a": NodeInfo(hostname="node-a", ranks=[0, 1], num_gpus=2, num_ranks=2),
            "node-b": NodeInfo(hostname="node-b", ranks=[2, 3], num_gpus=2, num_ranks=2),
        }

        return TopologyMap(
            world_size=4,
            num_nodes=2,
            gpu_info=gpus,
            nodes=nodes,
            fabric_domains={},
        )

    def test_no_fabric_all_rdma(self):
        topo = self._make_no_fabric_topology()
        assert topo.get_interconnect_level(0, 2) == InterconnectLevel.INTER_NODE_RDMA
        assert topo.get_interconnect_level(0, 1) == InterconnectLevel.INTRA_NODE

    def test_fabric_peers_empty(self):
        topo = self._make_no_fabric_topology()
        assert topo.get_fabric_domain_peers(0) == set()

    def test_comm_groups_no_fabric_is_empty(self):
        """With no fabric, fabric groups should be empty — not mirrored from intra-node."""
        topo = self._make_no_fabric_topology()
        td = TopologyDiscovery.__new__(TopologyDiscovery)
        td._topology = topo
        groups = td.get_communication_groups()
        assert groups[InterconnectLevel.INTRA_RACK_FABRIC] == []

    def test_comm_groups_intra_node_still_correct(self):
        """Intra-node groups are unaffected by the absence of fabric."""
        topo = self._make_no_fabric_topology()
        td = TopologyDiscovery.__new__(TopologyDiscovery)
        td._topology = topo
        groups = td.get_communication_groups()
        intra = groups[InterconnectLevel.INTRA_NODE]
        assert sorted(intra, key=lambda g: g[0]) == [[0, 1], [2, 3]]

    def test_comm_groups_rdma_still_covers_world(self):
        """RDMA group contains all ranks even when no fabric exists."""
        topo = self._make_no_fabric_topology()
        td = TopologyDiscovery.__new__(TopologyDiscovery)
        td._topology = topo
        groups = td.get_communication_groups()
        assert groups[InterconnectLevel.INTER_NODE_RDMA] == [[0, 1, 2, 3]]

    def test_heap_plan_no_fabric_peers(self):
        """With no fabric, heap plan should have empty fabric_peers for all ranks."""
        topo = self._make_no_fabric_topology()
        td = TopologyDiscovery.__new__(TopologyDiscovery)
        td._topology = topo
        plan = td.get_heap_distribution_plan()
        for rank, p in plan.items():
            assert p["fabric_peers"] == [], f"Rank {rank}: expected empty fabric_peers"
            assert p["fabric_domain"] == "", f"Rank {rank}: expected empty fabric_domain"


class TestAllFabricCluster:
    """Tests for a cluster where ALL nodes are in fabric domains."""

    def _make_all_fabric_topology(self):
        """
        node-a: ranks 0,1 (GPU0-1), fabric "aabb:0"
        node-b: ranks 2,3 (GPU0-1), fabric "aabb:0"
        """
        fabric = FabricInfo(cluster_uuid="aabb", clique_id=0)
        gpus = {}
        for r in range(2):
            gpus[r] = GPUInfo(
                global_rank=r,
                local_rank=r,
                hostname="node-a",
                gpu_id=r,
                pci_bus_id=f"0000:4{r}:00.0",
                device_name="MI300X",
                total_memory_mb=81920,
                numa_node=0,
                vendor="amd",
                uuid=f"gpu-a-{r}",
                fabric_info=fabric,
            )
        for r in range(2, 4):
            gpus[r] = GPUInfo(
                global_rank=r,
                local_rank=r - 2,
                hostname="node-b",
                gpu_id=r - 2,
                pci_bus_id=f"0000:8{r - 2}:00.0",
                device_name="MI300X",
                total_memory_mb=81920,
                numa_node=0,
                vendor="amd",
                uuid=f"gpu-b-{r - 2}",
                fabric_info=fabric,
            )

        nodes = {
            "node-a": NodeInfo(
                hostname="node-a",
                ranks=[0, 1],
                num_gpus=2,
                num_ranks=2,
                fabric_domain_key="aabb:0",
                fabric_domain_keys=["aabb:0"],
            ),
            "node-b": NodeInfo(
                hostname="node-b",
                ranks=[2, 3],
                num_gpus=2,
                num_ranks=2,
                fabric_domain_key="aabb:0",
                fabric_domain_keys=["aabb:0"],
            ),
        }

        return TopologyMap(
            world_size=4,
            num_nodes=2,
            gpu_info=gpus,
            nodes=nodes,
            fabric_domains={"aabb:0": ["node-a", "node-b"]},
        )

    def test_comm_groups_fabric_spans_nodes(self):
        """Fabric group merges ranks from both nodes."""
        topo = self._make_all_fabric_topology()
        td = TopologyDiscovery.__new__(TopologyDiscovery)
        td._topology = topo
        groups = td.get_communication_groups()
        fabric = groups[InterconnectLevel.INTRA_RACK_FABRIC]
        assert fabric == [[0, 1, 2, 3]]

    def test_comm_groups_no_standalone_groups(self):
        """When all nodes are in fabric, there should be no standalone groups."""
        topo = self._make_all_fabric_topology()
        td = TopologyDiscovery.__new__(TopologyDiscovery)
        td._topology = topo
        groups = td.get_communication_groups()
        fabric = groups[InterconnectLevel.INTRA_RACK_FABRIC]
        # Only one group — no standalone appendages
        assert len(fabric) == 1

    def test_comm_groups_intra_node_still_per_host(self):
        """Intra-node groups stay per-host even when fabric spans nodes."""
        topo = self._make_all_fabric_topology()
        td = TopologyDiscovery.__new__(TopologyDiscovery)
        td._topology = topo
        groups = td.get_communication_groups()
        intra = groups[InterconnectLevel.INTRA_NODE]
        assert sorted(intra, key=lambda g: g[0]) == [[0, 1], [2, 3]]

    def test_heap_plan_fabric_peers_cross_node(self):
        """Fabric peers should be cross-node ranks in the same domain."""
        topo = self._make_all_fabric_topology()
        td = TopologyDiscovery.__new__(TopologyDiscovery)
        td._topology = topo
        plan = td.get_heap_distribution_plan()
        # rank 0 on node-a: fabric peers = node-b ranks (cross-node, same domain)
        assert plan[0]["fabric_peers"] == [2, 3]
        assert plan[0]["ipc_peers"] == [1]
        assert plan[0]["rdma_peers"] == []

    def test_interconnect_cross_node_is_fabric(self):
        topo = self._make_all_fabric_topology()
        assert topo.get_interconnect_level(0, 2) == InterconnectLevel.INTRA_RACK_FABRIC
        assert topo.get_interconnect_level(1, 3) == InterconnectLevel.INTRA_RACK_FABRIC


# ---------------------------------------------------------------------------
# Distributed tests — require real GPUs and torchrun
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not dist.is_initialized(), reason="No distributed process group")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA/ROCm GPUs")
class TestDistributed:
    """Tests that run within a real distributed process group."""

    def test_all_gather_strings(self):
        """Test the string all-gather primitive."""
        rank = get_rank()
        world_size = get_world_size()
        local_str = f"hello_from_rank_{rank}"
        results = _all_gather_strings(local_str, world_size)
        assert len(results) == world_size
        for r in range(world_size):
            assert results[r] == f"hello_from_rank_{r}"

    def test_all_gather_strings_empty(self):
        world_size = get_world_size()
        results = _all_gather_strings("", world_size)
        assert results == [""] * world_size

    def test_all_gather_strings_large_payload(self):
        """Simulate large JSON payloads (a few KB each)."""
        rank = get_rank()
        world_size = get_world_size()
        payload = json.dumps({"rank": rank, "data": "x" * 4096})
        results = _all_gather_strings(payload, world_size)
        assert len(results) == world_size
        for r in range(world_size):
            parsed = json.loads(results[r])
            assert parsed["rank"] == r
            assert len(parsed["data"]) == 4096


@pytest.mark.skipif(not dist.is_initialized(), reason="No distributed process group")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA/ROCm GPUs")
class TestFullDiscovery:
    """End-to-end topology discovery tests."""

    def test_discover_returns_topology(self):
        td = TopologyDiscovery()
        topo = td.discover()
        assert isinstance(topo, TopologyMap)
        assert topo.world_size == get_world_size()
        assert topo.num_nodes >= 1

    def test_local_rank_is_unique_per_node(self):
        td = TopologyDiscovery()
        topo = td.discover()
        for hostname, node in topo.nodes.items():
            local_ranks = [topo.gpu_info[r].local_rank for r in node.ranks]
            assert len(set(local_ranks)) == len(local_ranks), f"Duplicate local_ranks on {hostname}: {local_ranks}"

    def test_own_rank_info_correct(self):
        td = TopologyDiscovery()
        topo = td.discover()
        rank = get_rank()
        info = topo.gpu_info[rank]
        assert info.global_rank == rank
        assert info.hostname == socket.gethostname()
        assert info.vendor in ("amd", "nvidia")
        assert info.total_memory_mb > 0
        assert info.pci_bus_id != ""

    def test_interconnect_symmetry(self):
        """Interconnect level should be symmetric: level(a,b) == level(b,a)."""
        td = TopologyDiscovery()
        topo = td.discover()
        ranks = sorted(topo.gpu_info.keys())
        for i, a in enumerate(ranks):
            for b in ranks[i + 1 :]:
                level_ab = topo.get_interconnect_level(a, b)
                level_ba = topo.get_interconnect_level(b, a)
                assert level_ab == level_ba, f"Asymmetric: level({a},{b})={level_ab} != level({b},{a})={level_ba}"

    def test_peer_partition_exhaustive(self):
        """For every rank, peers must partition the world."""
        td = TopologyDiscovery()
        topo = td.discover()
        world = set(range(get_world_size()))
        for rank in world:
            node = topo.get_node_peers(rank)
            fabric_only = topo.get_fabric_domain_peers(rank) - node
            rdma = topo.get_rdma_peers(rank)
            union = node | fabric_only | rdma | {rank}
            assert union == world, f"Rank {rank}: missing {world - union}"


class TestLogicalToPhysicalGpuIndex:
    """Tests for CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES index translation."""

    def test_no_env_var_returns_logical(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert _logical_to_physical_gpu_index(0, "nvidia") == 0
        assert _logical_to_physical_gpu_index(3, "nvidia") == 3

    def test_nvidia_remapping(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
        assert _logical_to_physical_gpu_index(0, "nvidia") == 2
        assert _logical_to_physical_gpu_index(1, "nvidia") == 3

    def test_amd_hip_visible(self, monkeypatch):
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "4,5,6")
        assert _logical_to_physical_gpu_index(0, "amd") == 4
        assert _logical_to_physical_gpu_index(2, "amd") == 6

    def test_amd_rocr_fallback(self, monkeypatch):
        monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1,3")
        assert _logical_to_physical_gpu_index(0, "amd") == 1
        assert _logical_to_physical_gpu_index(1, "amd") == 3

    def test_hip_takes_priority_over_rocr(self, monkeypatch):
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "7")
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "9")
        assert _logical_to_physical_gpu_index(0, "amd") == 7

    def test_logical_out_of_range_returns_logical(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
        # logical index 5 is beyond the 2-entry list
        assert _logical_to_physical_gpu_index(5, "nvidia") == 5

    def test_uuid_style_entry_returns_logical(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abcdef12-3456-7890")
        assert _logical_to_physical_gpu_index(0, "nvidia") == 0

    def test_negative_index_passthrough(self):
        assert _logical_to_physical_gpu_index(-1, "nvidia") == -1
