"""
Device-side tracing core: buffer allocation, capture, and export.
"""

import torch
import json
import pickle
import sys
import os
import socket

from .. import hip
from .events import EVENT_NAMES


class Tracing:
    """
    Manages device-side event tracing for an Iris instance.

    Handles trace buffer allocation, event capture, and export to Perfetto format.
    """

    def __init__(self, iris_instance):
        """
        Initialize tracing manager.

        Args:
            iris_instance: Parent Iris instance
        """
        self.iris = iris_instance
        self.enabled = False
        self.max_events = 0
        self.trace_buffers = {}
        self.trace_counter = None

    def enable(self, max_events=1_000_000):
        """
        Enable device-side event tracing.

        Allocates trace buffers to store events recorded by DeviceContext.

        Args:
            max_events (int): Maximum number of events to record. Default: 1,000,000
        """
        self.enabled = True
        self.max_events = max_events

        device = self.iris.device

        # Allocate trace buffers (Structure of Arrays for better memory access)
        self.trace_buffers = {
            "event_id": torch.zeros(max_events, dtype=torch.int32, device=device),
            "pid": torch.zeros(max_events, dtype=torch.int32, device=device),
            "pid_m": torch.zeros(max_events, dtype=torch.int32, device=device),
            "pid_n": torch.zeros(max_events, dtype=torch.int32, device=device),
            "cur_rank": torch.zeros(max_events, dtype=torch.int32, device=device),
            "target_rank": torch.zeros(max_events, dtype=torch.int32, device=device),
            "xcc_id": torch.zeros(max_events, dtype=torch.int32, device=device),
            "cu_id": torch.zeros(max_events, dtype=torch.int32, device=device),
            "timestamp": torch.zeros(max_events, dtype=torch.int64, device=device),
            "address": torch.zeros(max_events, dtype=torch.int64, device=device),
            "duration_cycles": torch.zeros(max_events, dtype=torch.int64, device=device),
        }

        # Atomic counter for event indexing
        self.trace_counter = torch.zeros(1, dtype=torch.int32, device=device)

        self.iris.info(f"Device tracing enabled with max {max_events} events")

    def reset(self):
        """
        Reset trace counter to start a new trace capture.

        Clears the event counter but keeps buffers allocated.
        """
        if not self.enabled:
            self.iris.warning("Tracing not enabled. Call tracing.enable() first.")
            return

        self.trace_counter.zero_()
        self.iris.debug("Trace buffers reset")

    def _collect_system_metadata(self):
        """Collect system and GPU metadata."""
        try:
            device_name = torch.cuda.get_device_name(self.iris.cur_rank)
        except Exception:
            device_name = "Unknown GPU"

        try:
            total_memory = torch.cuda.get_device_properties(self.iris.cur_rank).total_memory
            total_memory_gb = total_memory / (1024**3)
        except Exception:
            total_memory_gb = 0

        return {
            "process_name": os.path.basename(sys.argv[0]) if sys.argv else "unknown",
            "command_line": " ".join(sys.argv),
            "hostname": socket.gethostname(),
            "gpu_device_name": device_name,
            "gpu_total_memory_gb": f"{total_memory_gb:.2f}",
            "gpu_arch": hip.get_arch_string(self.iris.cur_rank),
            "gpu_cu_count": hip.get_cu_count(self.iris.cur_rank),
            "gpu_num_xcc": hip.get_num_xcc(self.iris.cur_rank),
            "rocm_version": hip.get_rocm_version(),
        }

    def _build_trace_events(self, num_events):
        """Build Perfetto trace events from captured data."""
        trace_events = []

        for i in range(num_events):
            event_id = int(self.trace_buffers["event_id"][i].item())
            event_name = EVENT_NAMES.get(event_id, f"unknown_{event_id}")

            pid = int(self.trace_buffers["pid"][i].item())
            cur_rank = int(self.trace_buffers["cur_rank"][i].item())
            target_rank = int(self.trace_buffers["target_rank"][i].item())
            xcc_id = int(self.trace_buffers["xcc_id"][i].item())
            cu_id = int(self.trace_buffers["cu_id"][i].item())
            begin_ts = int(self.trace_buffers["timestamp"][i].item())
            end_ts = int(self.trace_buffers["duration_cycles"][i].item())

            # Compute duration (0 = instant event)
            duration_cycles = (end_ts - begin_ts) if end_ts > 0 else 0

            # Perfetto event structure
            perfetto_event = {
                "name": event_name,
                "cat": "iris",
                "ts": begin_ts,
                "pid": cur_rank,
                "tid": f"XCC{xcc_id}_CU{cu_id}",
                "args": {
                    "program_id": pid,
                    "pid_m": int(self.trace_buffers["pid_m"][i].item()),
                    "pid_n": int(self.trace_buffers["pid_n"][i].item()),
                    "target_rank": target_rank,
                    "address": hex(int(self.trace_buffers["address"][i].item())),
                    "xcc_id": xcc_id,
                    "cu_id": cu_id,
                },
            }

            # Duration event or instant event?
            if duration_cycles > 0:
                perfetto_event["ph"] = "X"  # Complete event
                perfetto_event["dur"] = duration_cycles
            else:
                perfetto_event["ph"] = "i"  # Instant event
                perfetto_event["s"] = "t"

            trace_events.append(perfetto_event)

        # Add metadata event for this rank
        metadata = {
            "name": "process_name",
            "ph": "M",
            "pid": self.iris.cur_rank,
            "args": {"name": f"Rank {self.iris.cur_rank}"},
        }
        trace_events.append(metadata)

        return trace_events

    def export(self, filename="trace.json", merge=False):
        """
        Export collected trace events to Perfetto/Chrome Trace Event Format.

        All timestamps are in raw cycles from s_memrealtime (100MHz constant clock).
        View the output at: https://ui.perfetto.dev

        Args:
            filename (str): Output JSON filename. Default: "trace.json"
            merge (bool): If True, rank 0 collects and merges traces from all ranks
                         with timestamp alignment. If False, each rank exports its own file.

        Returns:
            dict: Trace data (merged on rank 0 if merge=True, per-rank otherwise)
        """
        import torch.distributed as dist

        if not self.enabled:
            self.iris.warning("Tracing not enabled. Call tracing.enable() first.")
            return {}

        # Get actual event count
        num_events = min(self.trace_counter.item(), self.max_events)

        # Collect metadata
        system_metadata = self._collect_system_metadata()

        # Build trace events
        trace_events = self._build_trace_events(num_events)

        # Write per-rank file
        per_rank_data = {
            "traceEvents": trace_events,
            "displayTimeUnit": "ns",
            "metadata": {
                "schema_version": "1.0",
                "num_events": num_events,
                "rank": self.iris.cur_rank,
                "world_size": self.iris.num_ranks,
                "time_unit": "raw cycles (s_memrealtime @ 100MHz)",
                **system_metadata,
            },
        }
        per_rank_filename = filename.replace(".json", f"_rank{self.iris.cur_rank}.json")
        with open(per_rank_filename, "w") as f:
            json.dump(per_rank_data, f, indent=2)
        self.iris.info(f"Exported rank {self.iris.cur_rank} trace to {per_rank_filename}")

        # If not merging, return per-rank data
        if not merge:
            return per_rank_data

        # Merging logic: serialize and gather events from all ranks
        events_bytes = pickle.dumps(trace_events)
        events_tensor = torch.ByteTensor(list(events_bytes)).cuda()

        # Gather event counts to rank 0
        event_counts = torch.tensor([len(events_bytes)], dtype=torch.int64, device="cuda")
        all_event_counts = [torch.zeros(1, dtype=torch.int64, device="cuda") for _ in range(self.iris.num_ranks)]
        dist.all_gather(all_event_counts, event_counts)

        # Synchronize before point-to-point communication to ensure proper ordering
        dist.barrier()

        # Rank 0: gather and merge all events
        if self.iris.cur_rank == 0:
            all_events = []

            for rank_id in range(self.iris.num_ranks):
                if rank_id == 0:
                    all_events.extend(trace_events)
                else:
                    recv_size = all_event_counts[rank_id].item()
                    recv_tensor = torch.zeros(recv_size, dtype=torch.uint8, device="cuda")
                    dist.recv(recv_tensor, src=rank_id)
                    recv_bytes = bytes(recv_tensor.cpu().numpy())
                    rank_events = pickle.loads(recv_bytes)
                    all_events.extend(rank_events)

            # Align timestamps: find minimum timestamp across all events
            all_timestamps = [e["ts"] for e in all_events if e.get("ph") != "M"]
            if all_timestamps:
                min_ts = min(all_timestamps)
                # Shift all timestamps to start from 0
                for event in all_events:
                    if event.get("ph") != "M":
                        event["ts"] = event["ts"] - min_ts

            merged_data = {
                "traceEvents": all_events,
                "displayTimeUnit": "ns",
                "metadata": {
                    "schema_version": "1.0",
                    "total_events": len(all_events),
                    "max_events": self.max_events,
                    "time_unit": "cycles (s_memrealtime @ 100MHz)",
                    "world_size": self.iris.num_ranks,
                    "timestamp_offset": min_ts if all_timestamps else 0,
                    "aligned": "minimum timestamp across all ranks",
                    **system_metadata,
                },
            }

            # Write merged file
            with open(filename, "w") as f:
                json.dump(merged_data, f, indent=2)

            self.iris.info(f"Exported {len(all_events)} merged trace events to {filename} (aligned)")
            self.iris.info("View at: https://ui.perfetto.dev")

            return merged_data
        else:
            # Other ranks: send events to rank 0
            dist.send(events_tensor, dst=0)
            return {}
