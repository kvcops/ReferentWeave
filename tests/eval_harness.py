import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from core.config import logger
from core.models import Chunk
from ingestion.resolver import CoreferenceResolver
from ingestion.enricher import enrich_chunk, enrich_document
from ingestion.parser import chunk_document_text, build_context_chunks, build_context_chunks_from_enriched
from ingestion.embedder import IndexingEngine
from retrieval.hybrid_search import HybridSearcher

# ---------------------------------------------------------------------------
# CROSS-CHUNK COREFERENCE BENCHMARK  — v3
# ---------------------------------------------------------------------------
# STRICT separation principle:
#
#   INTRO SECTION  (paras 1-10, entity named explicitly throughout)
#     → ONLY covers: architecture, specs, capabilities, performance, deployment.
#     → ZERO problem / failure / defect / solution content.
#
#   PRONOUN SECTION  (paras 11-22, entity name NEVER appears)
#     → ONLY covers: failure discovery, root cause, quantitative impact,
#       engineering fix, validation results, monitoring.
#     → ALL sentences use pronouns: it, they, these, this.
#
# Query design:
#   Query = entity name  +  specific problem/solution domain keyword
#   Answer snippet = numerical fact that lives ONLY in pronoun section
#
# Why Standard RAG fails on these queries:
#   BM25  → intro chunks rank high (entity name occurs 3-5× per chunk)
#            pronoun chunks rank low (entity name: 0 occurrences)
#   Dense → intro chunks rank low  (no problem content → bad semantic match)
#            pronoun chunks rank high (specific problem details → good match)
#   RRF   → intro chunks edge out pronoun chunks (BM25 dominates)
#         → top-1 = intro chunk → target snippet ABSENT → MISS
#
# Why ReferentWeave succeeds:
#   Resolver: identifies "'It'/'They' = <EntityName>" in pronoun section
#   Enricher: appends [Resolved Context Block] to every pronoun paragraph
#   BM25  → pronoun chunks NOW have entity name AND problem terms → rank HIGH
#   Dense → pronoun chunks still rank high (problem content + entity name)
#   RRF   → pronoun chunk clearly wins → top-1 = pronoun chunk
#         → target snippet PRESENT in target_text → HIT
# ---------------------------------------------------------------------------

EVAL_DATASET = [

    # =========================================================
    # DOCUMENT 1 : Prometheus-7 Neural Processing Unit
    # Domain      : AI Hardware / Data-Centre Accelerators
    # Intro       : pure specs/capabilities  (paras 1-10)
    # Pronoun     : thermal throttling crisis (paras 11-22)
    # =========================================================
    {
        "doc_id": "prometheus_7_npu.pdf",
        "full_text": (

            # ── INTRO : CAPABILITIES ONLY (entity name in every paragraph) ──

            "The Prometheus-7 Neural Processing Unit is CoreLogic Systems' flagship AI inference accelerator "
            "engineered for enterprise-scale transformer model serving at the edge. Announced at the 2025 "
            "Embedded AI Conference in San Francisco, the Prometheus-7 NPU targets deployments requiring "
            "sub-5 ms time-to-first-token latency guarantees with deterministic throughput under sustained "
            "concurrent request loads. CoreLogic Systems positioned the Prometheus-7 Neural Processing Unit "
            "as the successor to the Prometheus-4 series, incorporating three years of field data from "
            "over 12,000 deployed Prometheus-4 units across 47 hyperscaler and enterprise datacenter "
            "customers globally.\n\n"

            "The Prometheus-7 NPU is fabricated on Taiwan Semiconductor Manufacturing Company's 3-nanometer "
            "N3P process node, enabling a die area of 645 square millimeters and a transistor count of "
            "188 billion across a heterogeneous tile architecture. The Prometheus-7 Neural Processing Unit "
            "contains 72 configurable matrix engine tiles per die: 48 dedicated to self-attention head "
            "computation and 24 to feed-forward multilayer perceptron layers. Each attention tile on the "
            "Prometheus-7 NPU contains 128 systolic array matrix multiply accumulate units operating in "
            "FP8 precision at 2.4 gigahertz, yielding 2 teraFLOPS per tile and 96 teraFLOPS aggregate "
            "attention compute throughput across all 48 tiles.\n\n"

            "The Prometheus-7 Neural Processing Unit integrates 144 gigabytes of HBM3E memory across "
            "12 stacks bonded via through-silicon vias, delivering 10.2 terabytes per second of aggregate "
            "memory bandwidth. The Prometheus-7 NPU implements a hierarchical cache fabric consisting of "
            "a 768-megabyte global L3 cache, 64-megabyte per-tile L2 caches, and 4-megabyte L1 "
            "scratchpad memories co-located with each compute tile. This hierarchy enables the "
            "Prometheus-7 Neural Processing Unit to serve batch sizes of 128 concurrent requests at "
            "4,096-token context lengths without HBM3E bandwidth saturation under standard model weight "
            "distributions.\n\n"

            "The Prometheus-7 NPU employs the NeuroStream Scheduler, a custom dataflow orchestration "
            "engine that monitors per-tile utilization at 100-nanosecond intervals and dynamically "
            "reallocates tensor computation tasks to maintain load balance under bursty request arrivals. "
            "The NeuroStream Scheduler within the Prometheus-7 Neural Processing Unit uses a work-stealing "
            "algorithm adapted for spatial compute graphs, enabling idle tiles to proactively pull "
            "computation subgraphs from the global task queue. Across sustained LLaMA-3 70B inference "
            "benchmarks, the Prometheus-7 NPU consistently achieves 91% average tile utilization across "
            "all 48 attention tiles.\n\n"

            "The Prometheus-7 NPU interconnect fabric is a two-dimensional mesh connecting adjacent "
            "compute tiles at 56 gigabytes per second bidirectional bandwidth per link, enabling "
            "direct tile-to-tile tensor data sharing without routing through the L3 cache fabric. "
            "CoreLogic Systems integrated silicon photonic waveguides within the package substrate of "
            "the Prometheus-7 Neural Processing Unit to provide optical signaling between the compute "
            "die and the 12 HBM3E stacks at sub-400-picosecond latency per transaction, compared to "
            "the 3.2-nanosecond copper TSV latency in the Prometheus-4 series at equivalent bandwidth.\n\n"

            "The Prometheus-7 NPU implements hardware-enforced multi-tenant isolation, partitioning the "
            "tile array into isolated compute domains with per-tenant page table management and QoS "
            "arbitration in the NeuroStream Scheduler. The Prometheus-7 Neural Processing Unit ships "
            "with CoreLogic Systems' NeurOS SDK providing transparent API compatibility with PyTorch 2.4, "
            "JAX 0.4, and TensorFlow 2.17 through a CUDA-compatible device abstraction layer that "
            "eliminates code changes when migrating existing GPU-based inference pipelines.\n\n"

            "The Prometheus-7 NPU is rated at 520 watts thermal design power with a peak transient "
            "envelope of 580 watts. CoreLogic Systems designed the Prometheus-7 Neural Processing Unit "
            "power delivery network to sustain voltage integrity within ±2% across all compute tiles "
            "during simultaneous full-tile activation. The Prometheus-7 NPU ships in a dual-slot "
            "PCIe 5.0 x16 form factor with a 16-pin auxiliary power connector, available in "
            "air-cooled and direct liquid cooling configurations priced at $14,500 and $17,200 "
            "per unit respectively.\n\n"

            "Under the MLPerf Inference v5.0 benchmark suite, the Prometheus-7 NPU achieved "
            "4,200 tokens per second on LLaMA-3 70B with FP8 quantization at batch size 32 "
            "with a time-to-first-token of 2.8 milliseconds on 4,096-token prompts. The "
            "Prometheus-7 Neural Processing Unit ranked first in the MLPerf Datacenter Closed "
            "division for BERT-Large and GPT-J 6B, scoring 120,000 and 95,000 queries per second "
            "respectively, representing a 3.1× throughput improvement over the Prometheus-4 series.\n\n"

            "CoreLogic Systems commenced limited production of the Prometheus-7 NPU in Q1 2026 with "
            "initial delivery prioritized for four of the top-10 global cloud service providers. The "
            "Prometheus-7 Neural Processing Unit achieved UL 60950, CE Mark, and IEC 62368-1 "
            "certifications in January 2026, enabling shipments into North American and European "
            "data center facilities without additional regulatory qualification. General availability "
            "is targeted for Q3 2026 at volume pricing tiers of 50, 200, and 1,000-unit thresholds.\n\n"

            "Following the initial production deployment, CoreLogic Systems enrolled 847 Prometheus-7 "
            "NPU units across six enterprise datacenter facilities in the 90-day launch monitoring "
            "program. The Prometheus-7 Neural Processing Unit fleet achieved 92.6% of rated throughput "
            "across diverse deployed workloads and 99.91% fleet availability during the monitoring "
            "window, with downtime concentrated in three units experiencing host server PCIe controller "
            "firmware incompatibilities rather than NPU hardware defects. The NeurOS telemetry "
            "infrastructure collected performance counter, power, and temperature data from all "
            "847 deployed Prometheus-7 NPU units at 1-kilohertz polling rates throughout the period.\n\n"

            # ── PRONOUN SECTION : FAILURE ONLY (entity name NEVER appears below) ──

            "It encountered thermal throttling exclusively during inference workloads that combined "
            "batch sizes greater than 64 with sequence lengths exceeding 8,192 tokens, a pattern "
            "placing simultaneous peak demand on the attention tile array for KV-cache computation "
            "and on the HBM3E memory subsystem for key-value tensor retrieval. Approximately 23% of "
            "deployed units surfaced junction temperature readings above 80 degrees Celsius across "
            "the attention compute tile array under these workload conditions, well above the "
            "expected operating range for the rated thermal design. It was confirmed that the thermal "
            "anomalies were not present in laboratory qualification testing because those tests "
            "evaluated performance only under constant steady-state power loading rather than the "
            "dynamic thermal cycling patterns of real production inference traffic.\n\n"

            "They discovered through scanning electron microscopy on three returned field units "
            "that the thermal interface material applied between the silicon package lid and the "
            "aluminum heat spreader exhibited visible delamination and oxidation at the die-center "
            "region, corresponding precisely to the highest power density zone occupied by the "
            "16 central attention tiles. The delamination pattern was consistent with progressive "
            "thermal fatigue failure of the graphite-based thermal interface material under "
            "repeated power transients generated during attention head computation, which subject "
            "the material to temperature swings of 15-22 degrees Celsius at frequencies of "
            "2-8 kilohertz during sustained inference execution. These physical damage signatures "
            "were not captured by any accelerated aging qualification test in the original product "
            "design cycle, representing a significant gap in the thermal reliability testing protocol.\n\n"

            "It was subsequently determined through finite element thermal modeling calibrated "
            "against the scanning electron microscopy failure data that the thermal resistance "
            "of the degraded interface material had increased by an average of 0.18 degrees Celsius "
            "per watt compared to the as-manufactured specification, translating into a 32-degree "
            "Celsius increase in die-center junction temperature relative to the theoretical cooling "
            "capacity under the worst-case long-context inference workload. Specifically, it was "
            "established that the combination of increased thermal interface resistance and the "
            "die-center power density asymmetry causes a localized thermal gradient of 38 degrees "
            "Celsius between the die-center attention tile cluster and the die-peripheral "
            "feed-forward tiles during simultaneous sustained operation, a gradient that the "
            "original thermal design did not account for in its worst-case thermal budget.\n\n"

            "They began triggering the firmware-embedded thermal protection mechanism upon "
            "reaching junction temperatures of 85 degrees Celsius, which activates a graduated "
            "clock frequency reduction from the nominal 2.4 gigahertz to 2.0 gigahertz at "
            "85 degrees, 1.6 gigahertz at 90 degrees, and 1.1 gigahertz at 95 degrees Celsius. "
            "This clock frequency reduction translates to inference throughput degradation of "
            "16%, 33%, and 54% respectively at the three thresholds, with field telemetry "
            "confirming that the majority of throttling events sustained the 1.6 gigahertz "
            "reduced frequency for periods averaging 23 minutes. These throttling durations "
            "created measurable P99 inference latency spikes exceeding 45 milliseconds on "
            "affected serving endpoints, violating the enterprise SLA threshold of sub-5 ms "
            "time-to-first-token that the system was contracted to guarantee.\n\n"

            "They further revealed through spectral analysis of the throttling event logs that "
            "incidents exhibited a time-of-day pattern correlated with peak enterprise workload "
            "submission, with highest throttling frequency between 09:00 and 11:00 and again "
            "between 14:00 and 16:30 local datacenter time. It was additionally determined that "
            "throttling recovery hysteresis was substantially longer than anticipated, with units "
            "requiring an average of 11 minutes to recover to the 75-degree Celsius recovery "
            "threshold even after workload rates were reduced by 50%, because the increased "
            "thermal resistance of the degraded interface material slowed heat dissipation "
            "during the recovery phase. These recovery delays transformed episodic throttling "
            "events into sustained multi-hour throughput degradation windows during peak "
            "enterprise business hours.\n\n"

            "They were addressed through an engineering change order that redesigns the thermal "
            "interface material and heat spreading infrastructure, replacing the original "
            "graphite-based thermal interface pad with an indium-gallium eutectic alloy thermal "
            "interface material deposited using a controlled-atmosphere soldering process at "
            "157 degrees Celsius, forming a permanent metallurgical bond with both the silicon "
            "package lid and the vapor chamber heat spreader. This indium-gallium eutectic "
            "material maintains a thermal conductivity of 38.5 watts per meter-kelvin, compared "
            "to the original graphite pad's 15 watts per meter-kelvin initial conductivity and "
            "measured post-degradation conductivity of 12.3 watts per meter-kelvin, providing a "
            "3.1× improvement in thermal conductance at the die-to-heat-spreader interface. "
            "They also incorporate a vapor chamber heat spreader fabricated from sintered "
            "copper-titanium composite providing 2.8× superior lateral thermal spreading compared "
            "to the original solid copper design, redistributing die-center heat flux laterally.\n\n"

            "They must be installed by certified field service engineers using a specialized "
            "deposition tool controlling eutectic alloy uniformity to within ±5% across the die "
            "footprint to prevent void formation. This field replacement procedure takes "
            "approximately 35 minutes for the physical package swap and results in a complete "
            "maintenance window of 55 minutes per unit including power sequencing and firmware "
            "initialization. They are entitled to receive this engineering upgrade at no cost "
            "under the standard hardware warranty, which was extended from 3 years to 5 years "
            "for all customers affected by the thermal interface material field issue.\n\n"

            "They have been validated through accelerated thermal cycling qualification testing "
            "subjecting 24 engineering validation units to 15,000 thermal cycles between "
            "-40 degrees Celsius and +105 degrees Celsius under IEC 60068-2-14 thermal shock "
            "test standard, with no detectable change in thermal conductivity, bond adhesion "
            "strength, or junction temperature differential after completing the full sequence. "
            "This result corresponds to a predicted operational lifetime of greater than 12 years "
            "at the average field thermal cycling rate, providing a 2.4× margin over the "
            "5-year design service life target.\n\n"

            "They demonstrated in post-upgrade validation testing on 108 production units across "
            "three customer datacenter facilities that the new vapor chamber and indium-gallium "
            "eutectic thermal interface combination maintains peak die-center junction temperature "
            "below 67 degrees Celsius during the worst-case long-context high-batch inference "
            "workload that previously triggered throttling at 87 degrees Celsius, representing a "
            "20-degree Celsius improvement in thermal margin under identical workload conditions. "
            "It now sustains the rated 2.4 gigahertz tile clock frequency continuously without "
            "activating the thermal throttling protective mechanism under any validated inference "
            "workload configuration, restoring full rated throughput on LLaMA-3 70B benchmarks. "
            "These measurements were sustained across a 30-day post-upgrade monitoring window "
            "with zero throttling events recorded across all 108 upgraded units.\n\n"

            "They are additionally scheduled to receive a firmware update to version 3.4.1 that "
            "implements a predictive thermal headroom management algorithm, which uses on-die "
            "sensor data to forecast junction temperature trajectories 500 milliseconds in advance "
            "and applies preemptive workload throttling before hard throttling thresholds are "
            "reached. This predictive algorithm reduces effective peak junction temperature by an "
            "additional 4-6 degrees Celsius beyond the hardware upgrade benefit alone, providing a "
            "cumulative thermal headroom improvement of 24-26 degrees Celsius relative to "
            "unmodified units. They will receive this firmware automatically through the "
            "over-the-air update infrastructure beginning in the second week of March 2026.\n\n"

            "They will be enrolled in an enhanced fleet telemetry program that increases the "
            "on-die thermal sensor polling rate from 1 kilohertz to 10 kilohertz, transmitting "
            "10-millisecond-resolution junction temperature time series to the centralized fleet "
            "health monitoring service for early anomaly detection. This enhanced monitoring "
            "enables detection of thermal interface degradation onset within 7 days based on "
            "characteristic drift signatures in the high-resolution data, providing 3× earlier "
            "detection compared to the 21-day lag observed in the original field deployment "
            "where 1-second aggregates masked the early-stage degradation pattern."
        ),
        "queries": [
            {
                "query": "What thermal resistance increase caused the Prometheus-7 NPU throttling failure?",
                "target_snippet": "thermal resistance of the degraded interface material had increased by an average of 0.18 degrees Celsius per watt"
            },
            {
                "query": "What P99 latency violation did the Prometheus-7 NPU suffer during thermal throttling?",
                "target_snippet": "P99 inference latency spikes exceeding 45 milliseconds on affected serving endpoints"
            },
            {
                "query": "What replacement thermal interface material was used to fix the Prometheus-7 NPU?",
                "target_snippet": "indium-gallium eutectic alloy thermal interface material deposited using a controlled-atmosphere soldering process at 157 degrees Celsius"
            },
            {
                "query": "What junction temperature does the repaired Prometheus-7 NPU maintain under worst-case load?",
                "target_snippet": "peak die-center junction temperature below 67 degrees Celsius during the worst-case long-context high-batch inference workload"
            },
            {
                "query": "How long is the complete maintenance window for the Prometheus-7 NPU thermal upgrade?",
                "target_snippet": "complete maintenance window of 55 minutes per unit including power sequencing and firmware initialization"
            }
        ]
    },

    # =========================================================
    # DOCUMENT 2 : Aurora-Prime Quantum Computing Platform
    # Domain      : Quantum Computing / Superconducting Qubits
    # Intro       : pure specs/capabilities  (paras 1-10)
    # Pronoun     : qubit decoherence crisis (paras 11-21)
    # =========================================================
    {
        "doc_id": "aurora_prime_quantum.pdf",
        "full_text": (

            # ── INTRO : CAPABILITIES ONLY ──────────────────────────────────────

            "The Aurora-Prime Quantum Computing Platform is QuantumLogic Corporation's flagship "
            "1,024-qubit superconducting quantum processor, introduced at the International Quantum "
            "Hardware Summit in November 2025. The Aurora-Prime platform was designed to achieve "
            "quantum advantage on optimization and machine learning workloads by combining advances "
            "in transmon qubit fabrication, high-fidelity two-qubit gate implementation, and "
            "real-time classical control electronics developed over eight years of quantum hardware "
            "research at QuantumLogic Corporation.\n\n"

            "The Aurora-Prime Quantum Computing Platform implements 1,024 fixed-frequency transmon "
            "qubits fabricated on a 72mm×72mm niobium-on-sapphire substrate using a 7-layer "
            "superconducting process co-developed with IBM Research. Each Aurora-Prime transmon qubit "
            "is individually addressable through a coplanar waveguide resonator operating between "
            "4.8 and 5.3 gigahertz, with a cross-talk suppression ratio of -58 dB between adjacent "
            "qubit control lines enabling simultaneous single-qubit gate operations across the "
            "full 1,024-qubit array.\n\n"

            "The Aurora-Prime platform achieves single-qubit gate fidelities averaging 99.94% by "
            "randomized benchmarking across the full 1,024-qubit array, with the highest-fidelity "
            "qubits at 99.97% and the lowest-fidelity qubits at 99.89%. Two-qubit cross-resonance "
            "gate fidelities on the Aurora-Prime Quantum Computing Platform average 99.62% across "
            "all 2,048 nearest-neighbor qubit pairs in the heavy-hexagon connectivity topology, "
            "with less than 0.15% standard deviation in fidelity across the full array as measured "
            "during the 30-day commissioning characterization campaign.\n\n"

            "The Aurora-Prime Quantum Computing Platform implements a full-stack quantum error "
            "correction architecture supporting the surface code at code distances from d=3 to "
            "d=11, enabling logical qubit error rates of approximately 10^-6 per logical gate "
            "cycle at d=7 with physical error rates of 0.3% per two-qubit gate. QuantumLogic "
            "Corporation integrated 16 Xilinx Versal AI Core FPGAs into the Aurora-Prime classical "
            "co-processor subsystem, providing real-time syndrome decoding at the 1-microsecond "
            "cycle time demanded by transmon qubit T2 coherence times averaging 180 microseconds "
            "across the Aurora-Prime qubit array.\n\n"

            "The Aurora-Prime platform's cryogenic infrastructure maintains the 1,024-qubit "
            "processor at 8 millikelvin using a dilution refrigerator with 25 microwatts of "
            "cooling power at the mixing chamber stage, providing 3.1× thermal margin over the "
            "estimated heat load. QuantumLogic Corporation designed the Aurora-Prime cryogenic "
            "infrastructure to accommodate up to 2,048 qubits for a planned Aurora-Prime v2.0 "
            "scale-up within the existing refrigerator frame, reducing future expansion capital "
            "expenditure by an estimated 45%.\n\n"

            "QuantumLogic Corporation integrated the Aurora-Prime Quantum Computing Platform "
            "with their cloud infrastructure via a Qiskit-compatible circuit submission API "
            "supporting job priority queuing, batch circuit execution across multiple error "
            "correction code configurations, and automated calibration scheduling to maintain "
            "gate fidelities within 0.1% of specification across seven-day operational windows. "
            "The Aurora-Prime platform supports OpenQASM 3.0, Quil, and Cirq circuit formats "
            "through an automatic transpilation layer that optimizes native gate decompositions.\n\n"

            "QuantumLogic Corporation commercially deployed the Aurora-Prime Quantum Computing "
            "Platform in Q4 2025 across three facilities in New Haven, Connecticut; Zurich, "
            "Switzerland; and Tokyo, Japan. The Aurora-Prime platform received a certified "
            "quantum volume of 3,072 at commercial launch, independently verified by a quantum "
            "hardware characterization team from the National Institute of Standards and "
            "Technology. Subscription pricing starts at $15,000 per hour for dedicated "
            "1,024-qubit processing time.\n\n"

            "The Aurora-Prime Quantum Computing Platform supports three flagship application "
            "domains: pharmaceutical molecular simulation using variational quantum eigensolvers, "
            "financial portfolio optimization using the quantum approximate optimization "
            "algorithm, and logistics network planning using quantum-classical hybrid methods. "
            "QuantumLogic Corporation provides domain-specific quantum algorithm libraries "
            "pre-compiled for the Aurora-Prime native gate set and heavy-hexagon topology, "
            "reducing application development time by an estimated 60%.\n\n"

            "During the first 60 days of commercial operation, QuantumLogic Corporation "
            "processed 12,847 quantum circuit executions submitted by 23 enterprise customers. "
            "The Aurora-Prime Quantum Computing Platform achieved an average circuit success "
            "rate of 94.7% across all submitted circuits, measured as the percentage of shots "
            "returning the expected ground state bitstring on verification circuits included in "
            "each batch. Customer satisfaction scores averaged 4.4 out of 5.0 across post-job "
            "survey responses collected from enterprise account holders during the observation "
            "period.\n\n"

            "The Aurora-Prime platform earned FISMA Moderate authorization for US federal "
            "agency workloads in December 2025 and ISO 27001 certification for the New Haven "
            "facility in January 2026, enabling deployment for regulated healthcare and "
            "financial services quantum computing use cases. QuantumLogic Corporation operates "
            "the Aurora-Prime Quantum Computing Platform under a 99.9% monthly uptime SLA "
            "with scheduled maintenance windows of no more than 4 hours per calendar month, "
            "verified through independent uptime monitoring by a third-party SLA audit firm.\n\n"

            # ── PRONOUN SECTION : FAILURE ONLY (entity name NEVER appears below) ──

            "It began exhibiting qubit frequency drift events affecting 7.3% of the qubit "
            "array during the 60-day operational observation window, with affected qubits "
            "showing transition frequency shifts of 0.8-2.4 megahertz over 4-12 hour "
            "timescales relative to the calibrated resonance frequency values stored in the "
            "classical control electronics. These frequency shifts caused the two-qubit "
            "cross-resonance gates on affected qubit pairs to operate at increasingly "
            "off-resonance drive frequencies, systematically reducing gate fidelity below "
            "the specification required for surface code error correction at code distance "
            "d=7. It was identified through correlation analysis of telemetry records that "
            "the frequency drift rate was highest in the first 30 days of commercial "
            "operation and gradually stabilized over the following 30-day period, suggesting "
            "a burn-in phenomenon rather than a progressive hardware degradation.\n\n"

            "It was determined through a combination of impedance spectroscopy of the "
            "Josephson junction barrier and numerical modeling of charge noise spectral "
            "density that the primary source of the observed frequency drift was a "
            "0.3-nanometer increase in effective barrier oxide thickness occurring gradually "
            "over the first 1,000 operational hours as residual surface adsorbates diffused "
            "into the barrier under continuously applied microwave drive fields, causing a "
            "systematic upward shift of Josephson energy and qubit frequency at approximately "
            "180 kilohertz per week. It was further established that this barrier oxide "
            "stabilization process could be accelerated by applying an electromagnetic "
            "annealing protocol exposing each qubit to 500 coherent microwave pulses at its "
            "natural frequency, promoting surface adsorbate desorption without detectable "
            "bond restructuring in the barrier oxide crystal lattice.\n\n"

            "They exhibited frequency drift rates 3.7× higher before the electromagnetic "
            "annealing stabilization protocol was applied compared to after, with "
            "post-annealing qubits showing frequency drift rates below 40 kilohertz per "
            "week versus pre-annealing rates of 148 kilohertz per week on average across "
            "the affected qubit population. This drift rate reduction maintained two-qubit "
            "cross-resonance gate fidelities above 99.5% throughout the 30-day post-annealing "
            "observation period without requiring recalibration of gate drive frequencies, "
            "compared to the pre-annealing regime where recalibration was required every "
            "18 hours on average to maintain fidelities above 99.0%.\n\n"

            "They suffered concurrent gate fidelity degradation events at the New Haven "
            "facility that correlated with atmospheric pressure changes exceeding 2.5 "
            "millibars within 90-minute windows, which introduced microscopic mechanical "
            "stress variations in the dilution refrigerator's vibration isolation stack "
            "that transmitted acoustic coupling into the qubit substrate at frequencies "
            "overlapping the transmon qubit energy relaxation spectrum. These pressure-correlated "
            "events caused two-qubit gate fidelity to fall as low as 97.1% on the "
            "worst-affected qubit pairs during pressure excursion events, elevating the "
            "minimum required surface code distance from d=7 to d=9 for achieving the "
            "target logical error rate of 10^-6 per logical gate cycle.\n\n"

            "They were supplemented by installation of active vibration compensation "
            "actuators at the 4 kelvin cold plate stage that cancel measured vibration "
            "spectra from 0.1 to 50 hertz using a feed-forward algorithm driven by "
            "floor-mounted accelerometers, reducing coherent mechanical coupling from "
            "atmospheric pressure events by 28 dB across the critical 1-20 hertz range. "
            "This active vibration isolation completely eliminated the correlation between "
            "barometric pressure changes and two-qubit gate fidelity degradation, confirmed "
            "by 90 days of post-installation monitoring with zero pressure-correlated "
            "fidelity events recorded at any of the 30 monitored network qubit pairs.\n\n"

            "They resulted in a measured improvement in average two-qubit cross-resonance "
            "gate fidelity from the original 99.62% to 99.78% following implementation of "
            "both the electromagnetic annealing protocol and active vibration isolation, "
            "with the best-performing qubit pairs now achieving 99.91% and the "
            "lowest-performing pairs achieving 99.63% gate fidelity. This 0.16 percentage "
            "point average improvement reduces the surface code logical qubit error rate "
            "by 2.3× at code distance d=7, enabling logical error rates of 4.3×10^-7 per "
            "logical gate cycle and providing margin to reduce code distance to d=5 for "
            "circuit workloads tolerating 10^-5 logical error rate, achieving a 52% "
            "reduction in physical qubit overhead.\n\n"

            "They also demonstrated a 34% improvement in the quantum volume benchmark "
            "score, with the platform achieving a certified quantum volume of 4,096 "
            "compared to the pre-improvement certified quantum volume of 3,072, representing "
            "the highest certified quantum volume achieved by any commercially deployed "
            "quantum processor as of January 2026. It achieved this milestone using circuits "
            "of depth 12 and width 12, with the heavy output generation probability of "
            "0.512 exceeding the 2/3 quantum volume threshold with 5-standard-deviation "
            "statistical confidence across 5,000 circuit repetitions, independently "
            "certified by the National Institute of Standards and Technology.\n\n"

            "They are expected to enable quantum advantage over classical computing on "
            "combinatorial optimization problems in the 300-500 variable range based on "
            "algorithmic projections demonstrating theoretical speedups of 10×-100× over "
            "classical heuristic solvers using the improved two-qubit gate fidelities now "
            "available. It will begin accepting commercially submitted QUBO and MaxCut "
            "optimization circuit workloads from enterprise customers in Q2 2026 following "
            "system characterization for the new performance envelope, with subscription "
            "pricing adjusted to reflect the expanded quantum volume.\n\n"

            "They will also benefit from an update to the quantum error correction decoder "
            "running on the 16 Xilinx Versal FPGAs, replacing the minimum-weight perfect "
            "matching algorithm with a union-find decoder that processes syndrome measurement "
            "graphs 3.2× faster at code distances above d=7, enabling real-time error "
            "correction for deeper circuits without exceeding the 1-microsecond decoding "
            "budget. This decoder improvement extends the maximum circuit depth achievable "
            "at logical error rates below 10^-6 from 800 logical gate cycles to 1,900 "
            "logical gate cycles at code distance d=7, substantially broadening the range "
            "of quantum algorithms practical within the available coherence time budget.\n\n"

            "They continue to demonstrate stable operational performance through the "
            "February 2026 monitoring period, with 180-day fleet uptime reaching 99.94% "
            "across all three deployed facilities and zero new categories of hardware "
            "anomaly identified in the telemetry. It serves as the primary quantum "
            "computing resource for seven contracted enterprise customers actively "
            "developing quantum-classical hybrid algorithms, with three customers having "
            "published preliminary results demonstrating performance advantages over "
            "classical baseline solvers in their respective domain benchmark problem sets."
        ),
        "queries": [
            {
                "query": "What caused the qubit frequency drift degradation in the Aurora-Prime quantum processor?",
                "target_snippet": "0.3-nanometer increase in effective barrier oxide thickness occurring gradually over the first 1,000 operational hours"
            },
            {
                "query": "What were the drift rates before and after annealing stabilization in Aurora-Prime?",
                "target_snippet": "post-annealing qubits showing frequency drift rates below 40 kilohertz per week versus pre-annealing rates of 148 kilohertz per week"
            },
            {
                "query": "What improvement in two-qubit gate fidelity did Aurora-Prime achieve after fixes?",
                "target_snippet": "average two-qubit cross-resonance gate fidelity from the original 99.62% to 99.78%"
            },
            {
                "query": "What certified quantum volume score did Aurora-Prime achieve after the upgrades?",
                "target_snippet": "certified quantum volume of 4,096 compared to the pre-improvement certified quantum volume of 3,072"
            },
            {
                "query": "How many logical gate cycles can the Aurora-Prime decoder now support at low error rates?",
                "target_snippet": "maximum circuit depth achievable at logical error rates below 10^-6 from 800 logical gate cycles to 1,900 logical gate cycles"
            }
        ]
    },

    # =========================================================
    # DOCUMENT 3 : Helios-9 Smart Grid Control System
    # Domain      : Smart Grid / Distributed Energy Management
    # Intro       : pure specs/capabilities  (paras 1-10)
    # Pronoun     : grid oscillation crisis  (paras 11-21)
    # =========================================================
    {
        "doc_id": "helios_9_smart_grid.pdf",
        "full_text": (

            # ── INTRO : CAPABILITIES ONLY ──────────────────────────────────────

            "The Helios-9 Smart Grid Control System is GridTech Industries' flagship distributed "
            "energy resource management platform, coordinating 14,200 solar inverters, battery "
            "storage units, and demand response assets across the South Australian electricity "
            "network serving 1.8 million customers. The Helios-9 system represents GridTech "
            "Industries' fourth-generation grid management platform, building on the Helios-7 "
            "and Helios-8 deployments across four Australian state electricity networks between "
            "2021 and 2024.\n\n"

            "The Helios-9 Smart Grid Control System manages grid frequency regulation through a "
            "hierarchical control architecture: a central SCADA cluster in the GridTech Industries "
            "regional control center in Adelaide, 47 distribution automation controllers at zone "
            "substation level, and embedded firmware agents within each of the 14,200 managed "
            "assets. The central Helios-9 SCADA cluster operates on a 12-node server rack with "
            "480 teraflops of aggregate compute capacity, executing frequency deviation detection, "
            "economic dispatch optimization, and protective relay coordination at a "
            "100-millisecond control loop period.\n\n"

            "The Helios-9 Smart Grid Control System implements GridTech Industries' ElasticDispatch "
            "algorithm for real-time economic dispatch, simultaneously solving a 14,200-asset "
            "optimal power flow problem with 8,900 active constraints at 100-millisecond intervals "
            "using a distributed alternating direction method of multipliers across the 12-node "
            "cluster. This enables the Helios-9 system to dynamically reallocate reactive power "
            "injection setpoints across the managed inverter fleet in response to voltage deviation "
            "events at any of the 3,200 monitored grid nodes, achieving reactive power compensation "
            "response times of less than 250 milliseconds from deviation detection to fleet "
            "setpoint adjustment confirmation.\n\n"

            "GridTech Industries integrated the Helios-9 Smart Grid Control System with AEMO's "
            "Electricity Statement of Opportunities interface for automated frequency control "
            "ancillary service bid submission, enabling the managed fleet to participate in the "
            "5-minute settlement frequency regulation market with bid quantities up to 280 "
            "megawatts of combined upward and downward regulation capacity at 6-second response "
            "times meeting AEMO's contingency frequency regulation qualification requirement.\n\n"

            "The Helios-9 Smart Grid Control System implements real-time harmonic distortion "
            "compensation across the managed inverter fleet, continuously adjusting individual "
            "inverter pulse-width modulation switching patterns to cancel 5th, 7th, and 11th "
            "harmonic components generated by high-penetration photovoltaic inverter operation. "
            "GridTech Industries integrated a dedicated digital signal processing pipeline into "
            "the Helios-9 SCADA cluster for voltage total harmonic distortion measurement at "
            "all 3,200 monitored nodes using fast Fourier transform computation at "
            "10-millisecond window update rates.\n\n"

            "The Helios-9 system was deployed to the South Australian network in phases beginning "
            "August 2025, with full 14,200-asset enrollment completed December 2025 following "
            "commissioning testing of each distribution automation controller and end-to-end "
            "latency verification. The control communication infrastructure achieved 145-millisecond "
            "average round-trip latency for metropolitan Adelaide assets and 820-1,100-millisecond "
            "latency for remote Eyre Peninsula assets over the South Australian power utility's "
            "private fiber optic network.\n\n"

            "The Helios-9 Smart Grid Control System demonstrated a frequency deviation regulation "
            "response time of 4.2 seconds from onset to recovery to within ±0.05 hertz of the "
            "50-hertz nominal during a 1,100-megawatt generation trip event at Torrens Island, "
            "within AEMO's 5-second contingency response qualification requirement. The Helios-9 "
            "system achieved a 34% voltage quality improvement measured as reduction in voltage "
            "total harmonic distortion at 245 monitored distribution transformer secondary buses "
            "compared to the predecessor Helios-8 system baseline measurements.\n\n"

            "The Helios-9 Smart Grid Control System delivered 94.7% of bid quantity during "
            "frequency regulation ancillary service activations within the 6-second response "
            "window during the January 2026 assessment period, exceeding AEMO's 90% minimum "
            "qualification threshold. GridTech Industries' customers across the South Australian "
            "network reported a 22% reduction in voltage deviation event frequency and a 31% "
            "reduction in protection relay misoperation rate compared to the Helios-8 baseline "
            "during the first 30 days of Helios-9 operation.\n\n"

            "The Helios-9 Smart Grid Control System supports integration with 15 different "
            "distributed energy resource asset management platforms through IEC 61968-9 data "
            "exchange interfaces, enabling coordinated dispatch optimization across battery "
            "storage, demand response, and generation asset classes without proprietary "
            "integration middleware. GridTech Industries provides 24/7 system operations "
            "support for the Helios-9 platform from its Adelaide network operations center, "
            "staffed by teams of 8 certified network control engineers per shift.\n\n"

            "The Helios-9 Smart Grid Control System earned AEMO's System Strength Service "
            "Provider accreditation in November 2025, enabling the platform to provide "
            "compensated system strength services to the South Australian network service "
            "provider valued at an estimated AUD 12 million annually. GridTech Industries "
            "operates the Helios-9 platform under a 99.95% monthly availability SLA "
            "verified through independent uptime monitoring, with the central SCADA cluster "
            "achieving 100% availability throughout the Q4 2025 initial deployment phase.\n\n"

            # ── PRONOUN SECTION : FAILURE ONLY (entity name NEVER appears below) ──

            "It began exhibiting high-frequency voltage oscillations at 2.3-2.7 hertz "
            "affecting 12 distribution network areas following the commissioning of three "
            "gas turbine generation units at Pelican Point, Osborne, and Port Stanvac power "
            "stations, which replaced the prior synchronous condenser fleet retired in "
            "November 2025. These oscillations were captured in phasor measurement unit "
            "recordings at 30 network monitoring locations and oscilloscope waveform records "
            "from 12 distribution network areas, with voltage amplitude oscillations reaching "
            "1.8% of nominal voltage magnitude at the worst-affected monitoring sites. "
            "It also exhibited reactive power sharing instability events at three "
            "66-kilovolt bulk supply points causing power swings of up to 45 megavars at "
            "0.15-hertz frequency during midday peak photovoltaic generation periods.\n\n"

            "It was determined through sensitivity analysis of the control system eigenvalue "
            "model that the insufficient damping of the 2.3-hertz electromechanical oscillation "
            "mode was caused by an interaction between the proportional gain of the reactive "
            "power compensation control loop, which had been tuned for the synchronous condenser "
            "fleet present during commissioning, and the substantially lower inertia constant "
            "of the replacement gas turbine generation units. Specifically, it was established "
            "that the reactive power control loop gain of 45 megavars per hertz, which provided "
            "7.3% modal damping with synchronous condenser inertia, produced only 2.1% modal "
            "damping with the gas turbine inertia constants of 2.1-2.8 seconds, requiring "
            "a gain reduction to 28 megavars per hertz to restore the oscillation mode damping "
            "ratio to the AEMO-required 5.0% minimum threshold.\n\n"

            "They were implemented as a firmware parameter update to all 47 distribution "
            "automation controllers, adjusting the reactive power compensation control loop "
            "gain from 45 to 28 megavars per hertz and simultaneously increasing derivative "
            "control gain from 2.1 to 3.4 megavars per hertz-squared. This firmware update "
            "was deployed over a 4-hour window on February 8, 2026 with staged rollout "
            "beginning at 02:00 ACST to minimize disruption during the network's "
            "lowest-demand period. They were applied without any service interruption to "
            "frequency regulation ancillary service delivery, with continuous phasor "
            "measurement monitoring confirming elimination of the 2.3-hertz oscillation "
            "within 90 seconds of each controller completing the update.\n\n"

            "They resulted in a measured improvement in the 2.3-hertz electromechanical "
            "oscillation mode damping ratio from 2.1% to 6.8% across the full "
            "47-controller fleet following the firmware gain parameter update, verified "
            "through 72-hour post-update phasor measurement unit recording analysis showing "
            "no recurrence of oscillation events with amplitudes exceeding 0.5% of nominal "
            "voltage magnitude. It concurrently demonstrated a 41% reduction in peak "
            "reactive power rate of change at the three affected bulk supply points, "
            "reducing the measured maximum from 18 megavars per second to 10.6 megavars "
            "per second, within the 12 megavar per second design specification limit.\n\n"

            "They also demonstrated a secondary benefit from the reactive power coordination "
            "algorithm improvement addressing the sharing instability events, implementing "
            "a consensus-based protocol synchronizing reactive power setpoint updates between "
            "battery storage and photovoltaic inverter clusters using a distributed averaging "
            "algorithm converging to a globally consistent allocation within 3 control loop "
            "cycles of 300 milliseconds. This coordination improvement eliminated the limit "
            "cycle oscillations completely, reducing peak reactive power swings at the "
            "three bulk supply points from 45 megavars to less than 8 megavars during "
            "highest-irradiance midday conditions, an 82% reduction in reactive power "
            "swing amplitude without modifying the ElasticDispatch economic dispatch "
            "objective function.\n\n"

            "They have been validated through a post-improvement assessment in February 2026 "
            "encompassing 28 days of continuous phasor measurement unit monitoring, analysis "
            "of 156 frequency regulation activation events responding to actual network "
            "contingencies, and deliberate test disturbance injections at three authorized "
            "network test points. This assessment confirmed average frequency regulation bid "
            "delivery of 96.3% of committed quantity within the 6-second response window, "
            "a 1.6 percentage point improvement over the January 2026 baseline before the "
            "oscillation fix was applied. It recorded zero voltage oscillation events "
            "exceeding the 0.5% amplitude threshold and zero reactive power rate of change "
            "events exceeding 12 megavars per second during the entire 28-day "
            "post-improvement assessment period.\n\n"

            "They are planning deployment of the Helios-10 enhancement module in Q3 2026, "
            "adding artificial intelligence-based predictive voltage stability monitoring "
            "using long short-term memory neural networks trained on 24 months of historical "
            "phasor measurement unit recordings, enabling anticipation of voltage stability "
            "margin degradation events up to 45 minutes before onset. This predictive "
            "capability will enable proactive reactive power setpoint adjustments across "
            "the managed fleet to restore voltage stability margins before emergency "
            "frequency control interventions become necessary, reducing emergency scheme "
            "activations by an estimated 67% based on retrospective analysis of historical "
            "South Australian network events.\n\n"

            "They are expected to increase the South Australian network's renewable energy "
            "instantaneous hosting capacity from 87% to greater than 95% renewable "
            "penetration following the Helios-10 predictive stability enhancement, enabling "
            "full decarbonization of South Australian electricity supply during "
            "high-irradiance solar generation periods. This improvement corresponds to an "
            "estimated 2.1 million tonnes of carbon dioxide equivalent emissions reduction "
            "annually from gas-fired generation that would otherwise be dispatched to "
            "maintain voltage stability margins during high renewable penetration operating "
            "conditions.\n\n"

            "They continue to meet all AEMO operational performance requirements through "
            "the Q1 2026 compliance reporting period, with the full 14,200-asset fleet "
            "maintaining aggregate uptime of 99.89% and the central SCADA cluster "
            "achieving 100% availability throughout the reporting period. It provides "
            "continuous IEC 61968-9 data exchange to the South Australian network service "
            "provider's energy management system at 1-second update rates for all "
            "14,200 managed asset operational state parameters, enabling real-time "
            "integrated network management without proprietary middleware.\n\n"

            "They are also scheduled to receive a firmware update implementing a "
            "model predictive control layer that uses real-time weather forecast data "
            "and historical solar irradiance patterns to pre-position battery storage "
            "state-of-charge across the managed fleet 15 minutes before anticipated "
            "generation ramp events, reducing the reactive power response burden on "
            "inverter-based resources during rapid solar generation changes. This model "
            "predictive control layer is expected to reduce the peak reactive power demand "
            "on the managed fleet during morning solar ramp-up events by 31%, substantially "
            "improving frequency regulation headroom during the highest-risk grid "
            "operating periods of the South Australian summer."
        ),
        "queries": [
            {
                "query": "What caused the 2.3 hertz voltage oscillation instability in the Helios-9 smart grid?",
                "target_snippet": "interaction between the proportional gain of the reactive power compensation control loop"
            },
            {
                "query": "What specific gain reduction was required to fix the Helios-9 oscillation mode?",
                "target_snippet": "gain reduction to 28 megavars per hertz to restore the oscillation mode damping ratio"
            },
            {
                "query": "What oscillation damping ratio did the Helios-9 system achieve after the firmware update?",
                "target_snippet": "2.3-hertz electromechanical oscillation mode damping ratio from 2.1% to 6.8%"
            },
            {
                "query": "How much did the Helios-9 fix reduce reactive power swings at the bulk supply points?",
                "target_snippet": "reducing peak reactive power swings at the three bulk supply points from 45 megavars to less than 8 megavars"
            },
            {
                "query": "What frequency regulation bid delivery rate did the Helios-9 system achieve after fixes?",
                "target_snippet": "average frequency regulation bid delivery of 96.3% of committed quantity within the 6-second response window"
            }
        ]
    }
]


def run_evaluation():
    logger.info("Initializing Cross-Chunk Coreference Evaluation Harness v3...")

    import shutil
    test_index_dir = Path("./data/test_index")
    if test_index_dir.exists():
        shutil.rmtree(test_index_dir)
    test_index_dir.mkdir(parents=True, exist_ok=True)

    try:
        resolver = CoreferenceResolver()

        # ── System 1: Standard RAG (no coreference enrichment) ────────────────
        logger.info("\n=== System 1: Standard RAG (Ablation) ===")
        standard_indexer = IndexingEngine(index_dir=test_index_dir / "standard")
        std_chunks = []
        ctr = 5000
        for doc in EVAL_DATASET:
            raw_splits = chunk_document_text(doc["full_text"])
            doc_chunks = build_context_chunks(raw_splits, doc["doc_id"] + "_std")
            for chunk in doc_chunks:
                chunk.chunk_id = ctr
                chunk.enriched_text = chunk.target_text   # no enrichment
                std_chunks.append(chunk)
                ctr += 1
        logger.info(f"Standard RAG: {len(std_chunks)} total chunks across {len(EVAL_DATASET)} documents")
        standard_indexer.ingest_chunks(std_chunks)
        standard_searcher = HybridSearcher(standard_indexer)

        # ── System 2: ReferentWeave RAG (full document-wide resolver) ─────────
        logger.info("\n=== System 2: ReferentWeave RAG ===")
        rw_indexer = IndexingEngine(index_dir=test_index_dir / "referentweave")
        rw_chunks = []
        ctr = 7000
        for doc in EVAL_DATASET:
            logger.info(f"  Resolving coreferences: {doc['doc_id']}")
            res_response = resolver.resolve_document(doc["full_text"])
            enriched_text = enrich_document(doc["full_text"], res_response.resolutions)
            raw_splits = chunk_document_text(enriched_text)
            doc_chunks = build_context_chunks_from_enriched(raw_splits, doc["doc_id"] + "_rw")
            for chunk in doc_chunks:
                chunk.chunk_id = ctr
                rw_chunks.append(chunk)
                ctr += 1
        logger.info(f"ReferentWeave: {len(rw_chunks)} total chunks across {len(EVAL_DATASET)} documents")
        rw_indexer.ingest_chunks(rw_chunks)
        rw_searcher = HybridSearcher(rw_indexer)

        # ── Evaluation ────────────────────────────────────────────────────────
        logger.info("\n=== Running Evaluation Queries ===")
        total_queries = 0
        std_success = 0
        rw_success = 0
        rows = []

        for doc in EVAL_DATASET:
            for q_info in doc["queries"]:
                total_queries += 1
                query = q_info["query"]
                snippet = q_info["target_snippet"]

                std_res = standard_searcher.search(query, doc_ids=[doc["doc_id"] + "_std"])
                std_hit = bool(std_res and snippet in std_res[0].text)
                if std_hit:
                    std_success += 1

                rw_res = rw_searcher.search(query, doc_ids=[doc["doc_id"] + "_rw"])
                rw_hit = bool(rw_res and snippet in rw_res[0].text)
                if rw_hit:
                    rw_success += 1

                rows.append({
                    "doc": doc["doc_id"],
                    "query": query,
                    "std_hit": "YES" if std_hit else "NO",
                    "rw_hit": "YES" if rw_hit else "NO",
                })

        # ── Print Results ─────────────────────────────────────────────────────
        W = 106
        print("\n" + "=" * W)
        print("REFERENTWEAVE — CROSS-CHUNK COREFERENCE RECALL  (Recall@1, End-to-End Pipeline)")
        print("=" * W)
        print(f"{'Document':<36} | {'Query':<49} | {'Std RAG':<7} | {'RW'}")
        print("-" * W)
        for r in rows:
            q = (r["query"][:47] + "..") if len(r["query"]) > 49 else r["query"]
            print(f"{r['doc']:<36} | {q:<49} | {r['std_hit']:<7} | {r['rw_hit']}")
        print("-" * W)

        std_r = (std_success / total_queries) * 100
        rw_r  = (rw_success  / total_queries) * 100
        gain  = rw_r - std_r
        print(f"{'OVERALL RECALL @ 1':<88} | {std_r:>5.1f}%  | {rw_r:.1f}%")
        print(f"{'ABSOLUTE GAIN  (ReferentWeave - Standard RAG)':<88} | {'':>7} | +{gain:.1f}%")
        print("=" * W)
        print(f"\nTotal queries  : {total_queries}")
        print(f"Standard RAG   : {std_success}/{total_queries} ({std_r:.1f}%)")
        print(f"ReferentWeave  : {rw_success}/{total_queries}  ({rw_r:.1f}%)")

    finally:
        import shutil as _sh
        if test_index_dir.exists():
            try:
                _sh.rmtree(test_index_dir)
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")


if __name__ == "__main__":
    run_evaluation()
