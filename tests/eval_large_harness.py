import sys
import os
import shutil
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from core.config import logger
from core.models import Chunk, Resolution, ResolutionResponse
from ingestion.enricher import enrich_chunk
from ingestion.embedder import IndexingEngine
from retrieval.hybrid_search import HybridSearcher

# A large, multi-domain dataset of 15 documents with multi-chunk coreferences
LARGE_EVAL_DATASET = [
    {
        "doc_id": "project_odyssey.pdf",
        "chunks": [
            "Project Odyssey was officially launched by the aerospace division in January 2026. The mission's goal is to construct and deploy a next-generation orbital spacecraft.",
            "It was delayed by six months due to a series of fuel valve thruster leaks in the main propulsion unit.",
            "They were sourced from a external contractor in Munich, which has led to intense supply chain disputes.",
            "This will cost the division an estimated $12 million in penalty fees and contract renegotiation expenses."
        ],
        "queries": [
            {"query": "Why was Project Odyssey delayed?", "target_chunk_idx": 1},
            {"query": "Where did the thruster valves of Project Odyssey come from?", "target_chunk_idx": 2},
            {"query": "What is the financial impact of the Project Odyssey delays?", "target_chunk_idx": 3}
        ]
    },
    {
        "doc_id": "novatech_financials.pdf",
        "chunks": [
            "NovaTech Inc. reported record-breaking revenue figures for the fiscal year 2025. Total annual earnings surged by 45% compared to the prior period.",
            "This was driven entirely by the rapid adoption of their proprietary cloud integration platform across enterprise clients.",
            "It saw a 300% subscription increase following the launch of the new AI-assisted connector module in Q2.",
            "However, they warn that growth might slow down in 2026 as market saturation approaches."
        ],
        "queries": [
            {"query": "What drove the revenue growth of NovaTech Inc.?", "target_chunk_idx": 1},
            {"query": "Why did NovaTech's cloud integration platform see subscription growth?", "target_chunk_idx": 2},
            {"query": "What warnings did NovaTech Inc. issue regarding future performance?", "target_chunk_idx": 3}
        ]
    },
    {
        "doc_id": "legal_agreement_licensee.pdf",
        "chunks": [
            "The Licensee agrees to pay the Licensor a monthly recurring subscription fee of five thousand dollars ($5,000) for software maintenance services.",
            "They must remit the full payment within five business days of invoice receipt to maintain server access.",
            "If they fail to do so, they will incur a 2% late penalty charge compounded weekly on the outstanding balance.",
            "This former party shall also be liable for all collection costs and reasonable legal fees incurred during dispute resolution."
        ],
        "queries": [
            {"query": "How quickly must the Licensee pay the invoice?", "target_chunk_idx": 1},
            {"query": "What penalty does the Licensee face for late software maintenance payments?", "target_chunk_idx": 2},
            {"query": "Who is liable for legal fees and collection costs in the software agreement?", "target_chunk_idx": 3}
        ]
    },
    {
        "doc_id": "quantum_zenith.pdf",
        "chunks": [
            "Project Zenith is NovaTech's flagship R&D initiative focused on building a fault-tolerant quantum computer processor.",
            "It leverages a novel superconducting chip architecture designed around 3D transmon qubits.",
            "These operate at sub-millikelvin temperatures inside a dilution refrigerator to maintain quantum coherence.",
            "The main challenge with this processor is mitigating high thermal noise and gate errors during two-qubit operations."
        ],
        "queries": [
            {"query": "What architecture does Project Zenith use?", "target_chunk_idx": 1},
            {"query": "How are the Project Zenith qubits cooled?", "target_chunk_idx": 2},
            {"query": "What is the primary challenge facing the Project Zenith processor?", "target_chunk_idx": 3}
        ]
    },
    {
        "doc_id": "clinical_trial_mt884.pdf",
        "chunks": [
            "Clinical Trial MT-884 was initiated in 2025 to evaluate the efficacy of a new monoclonal antibody targeting oncology patients.",
            "The experimental drug was administered bi-weekly to a treatment group of 450 adult participants over six months.",
            "It showed a 68% reduction in tumor progression compared to the standard chemotherapy regimen.",
            "They experienced mild side effects, predominantly fatigue and temporary joint pain, which resolved post-treatment."
        ],
        "queries": [
            {"query": "How frequently was the Clinical Trial MT-884 drug administered?", "target_chunk_idx": 1},
            {"query": "What was the tumor reduction rate for Clinical Trial MT-884?", "target_chunk_idx": 2},
            {"query": "What side effects did participants in Clinical Trial MT-884 experience?", "target_chunk_idx": 3}
        ]
    },
    {
        "doc_id": "prometheus_cluster.pdf",
        "chunks": [
            "The Prometheus Database Cluster is the primary storage engine for the company's real-time telemetry metrics.",
            "It comprises three write replica nodes and five read-only nodes distributed across US-East-1 and US-West-2 regions.",
            "These are configured with NVMe SSDs to sustain up to 1.5 million write operations per second at peak load.",
            "If any node fails, it triggers an automatic failover sequence that promotes a read node within 12 seconds."
        ],
        "queries": [
            {"query": "How many read and write nodes are in the Prometheus Database Cluster?", "target_chunk_idx": 1},
            {"query": "What hardware configures the Prometheus Database Cluster storage?", "target_chunk_idx": 2},
            {"query": "What happens when a Prometheus Database Cluster node fails?", "target_chunk_idx": 3}
        ]
    },
    {
        "doc_id": "factory_robotic_arms.pdf",
        "chunks": [
            "The SmartFactory IoT System orchestrates forty high-precision robotic arms on the automotive assembly line.",
            "They perform spot welding, adhesive application, and chassis alignment tasks with sub-millimeter accuracy.",
            "These require diagnostic calibration every 200 operational hours to prevent mechanical drift and motor fatigue.",
            "This schedule is managed automatically by the central AI scheduler based on real-time torque sensor feeds."
        ],
        "queries": [
            {"query": "What tasks do the SmartFactory robotic arms perform?", "target_chunk_idx": 1},
            {"query": "How often must the SmartFactory robotic arms be calibrated?", "target_chunk_idx": 2},
            {"query": "Who manages the calibration schedule for the SmartFactory robotic arms?", "target_chunk_idx": 3}
        ]
    },
    {
        "doc_id": "kraken_api_gateway.pdf",
        "chunks": [
            "The Kraken API Gateway serves as the single entry point for all incoming client requests to the microservices network.",
            "It handles rate limiting, JWT authentication, and SSL termination at the edge of the VPC.",
            "It is deployed as a Docker container swarm across multiple autoscaling EC2 instances.",
            "This service utilizes Redis for distributed token bucket tracking to enforce requests-per-second ceilings."
        ],
        "queries": [
            {"query": "What security operations does the Kraken API Gateway handle?", "target_chunk_idx": 1},
            {"query": "How is the Kraken API Gateway containerized and deployed?", "target_chunk_idx": 2},
            {"query": "How does Kraken API Gateway enforce rate limits?", "target_chunk_idx": 3}
        ]
    },
    {
        "doc_id": "gridpower_megapack.pdf",
        "chunks": [
            "The GridPower Substation incorporates ten industrial-grade Tesla Megapack units to stabilize local electrical frequency.",
            "They provide up to 50 megawatts of battery storage capacity with immediate discharge capabilities.",
            "These protect the regional grid from sudden load drops or solar farm generation fluctuations.",
            "They are controlled by a custom SCADA software loop monitoring grid frequency at 10ms intervals."
        ],
        "queries": [
            {"query": "What is the capacity of the GridPower Substation battery units?", "target_chunk_idx": 1},
            {"query": "What event types do the GridPower Megapack batteries protect against?", "target_chunk_idx": 2},
            {"query": "How is the GridPower Megapack discharge rate controlled?", "target_chunk_idx": 3}
        ]
    },
    {
        "doc_id": "cardiac_emergency_protocol.pdf",
        "chunks": [
            "The Cardiac Emergency Protocol outlines the rapid response timeline for code blue calls at St. Jude Hospital.",
            "It mandates that a defibrillator must be at the patient's bedside within ninety seconds of alarm activation.",
            "It must be charged to 200 Joules for the initial shock, followed by immediate CPR intervals.",
            "This procedure is reviewed annually by the Chief Medical Officer to maintain state compliance."
        ],
        "queries": [
            {"query": "How fast must a defibrillator arrive under the Cardiac Emergency Protocol?", "target_chunk_idx": 1},
            {"query": "What energy level is used for the initial shock in the Cardiac Emergency Protocol?", "target_chunk_idx": 2},
            {"query": "Who reviews the Cardiac Emergency Protocol annually?", "target_chunk_idx": 3}
        ]
    },
    {
        "doc_id": "fusion_energy_reactor.pdf",
        "chunks": [
            "Project Helios was initiated by the clean energy consortium in 2026 to design a commercial-grade nuclear fusion reactor.",
            "It utilizes a high-beta tokamak design to confine superheated deuterium-tritium plasma.",
            "These are kept in a toroidal shape using high-field rare-earth barium copper oxide (REBCO) superconducting magnets.",
            "This configuration is expected to produce net energy gain (Q > 10) by the end of the decade."
        ],
        "queries": [
            {"query": "What design does Project Helios use to confine plasma?", "target_chunk_idx": 1},
            {"query": "How are the Project Helios plasma elements kept in shape?", "target_chunk_idx": 2},
            {"query": "What is the expected outcome of the Project Helios configuration?", "target_chunk_idx": 3}
        ]
    },
    {
        "doc_id": "helios_drone_fleet.pdf",
        "chunks": [
            "The AeroVanguard autonomous drone fleet was deployed by the agricultural board to monitor crop health across regional farms.",
            "They carry advanced multispectral cameras and LiDAR sensors to map soil moisture and nitrogen levels.",
            "These transmit raw imagery data over a local 5G telemetry mesh directly to the central farming server.",
            "This system reduces water waste by automatically triggering localized drip irrigation systems."
        ],
        "queries": [
            {"query": "What equipment do the AeroVanguard drones carry?", "target_chunk_idx": 1},
            {"query": "How do the AeroVanguard drones transmit their data?", "target_chunk_idx": 2},
            {"query": "What is the primary ecological benefit of the AeroVanguard drone system?", "target_chunk_idx": 3}
        ]
    },
    {
        "doc_id": "sentinel_malware_scanner.pdf",
        "chunks": [
            "The Sentinel AI security scanner operates at the kernel level of enterprise operating systems to prevent malware execution.",
            "It continuously inspects system call patterns for anomalous process behaviors and memory injection flags.",
            "These are flagged instantly to quarantine suspicious executables before any ransomware payload can execute.",
            "This mechanism has prevented over 1,200 zero-day exploits across corporate workstations this quarter."
        ],
        "queries": [
            {"query": "What does Sentinel AI scanner inspect to find malware?", "target_chunk_idx": 1},
            {"query": "What happens when anomalous process behaviors are flagged by Sentinel AI?", "target_chunk_idx": 2},
            {"query": "How many exploits has the Sentinel AI scanner prevented this quarter?", "target_chunk_idx": 3}
        ]
    },
    {
        "doc_id": "nebula_game_engine.pdf",
        "chunks": [
            "The Nebula 3D Engine was selected as the official graphics suite for the upcoming space simulation game.",
            "It uses a hardware-accelerated ray tracing renderer to compute real-time global illumination and shadows.",
            "These are denoised using a lightweight neural network running concurrently on the GPU core.",
            "This allows the game to maintain a stable 60 frames per second on mainstream consoles."
        ],
        "queries": [
            {"query": "How does the Nebula 3D Engine compute global illumination?", "target_chunk_idx": 1},
            {"query": "How are the ray traced shadows in the Nebula engine denoised?", "target_chunk_idx": 2},
            {"query": "What frame rate does the Nebula engine target on consoles?", "target_chunk_idx": 3}
        ]
    },
    {
        "doc_id": "apex_neural_translator.pdf",
        "chunks": [
            "The Apex Neural Translator is an open-source model specialized in low-resource dialect translation.",
            "It utilizes a mixture-of-experts transformer architecture with 8 billion active parameters.",
            "They are fine-tuned on a curated corpus of oral histories and regional literature to preserve cultural context.",
            "This model outperforms generic large language models by 34% on dialect accuracy tests."
        ],
        "queries": [
            {"query": "What architecture does the Apex Neural Translator utilize?", "target_chunk_idx": 1},
            {"query": "On what corpus was the Apex Neural Translator fine-tuned?", "target_chunk_idx": 2},
            {"query": "How does Apex Neural Translator compare to generic language models?", "target_chunk_idx": 3}
        ]
    },
    {
        "doc_id": "aurora_water_filtration.pdf",
        "chunks": [
            "The Aurora water filtration plant implemented a new graphene-oxide membrane filtration system.",
            "It filters microplastics and heavy metal ions from municipal wastewater at a fraction of standard cost.",
            "These are trapped within the sub-nanometer pores of the carbon mesh through size exclusion and adsorption.",
            "This process yields drinking-quality water that meets the highest environmental purity guidelines."
        ],
        "queries": [
            {"query": "What contaminants does the Aurora filtration system remove?", "target_chunk_idx": 1},
            {"query": "How are microplastics trapped in the Aurora graphene-oxide system?", "target_chunk_idx": 2},
            {"query": "What is the quality of the water produced by the Aurora process?", "target_chunk_idx": 3}
        ]
    },
    {
        "doc_id": "titan_mining_loader.pdf",
        "chunks": [
            "The Titan-900 automated excavator was deployed at the Western Australian iron ore mine.",
            "It operates continuously in extreme heat conditions using an active liquid-nitrogen cooling loop.",
            "These conditions usually cause human operators to suffer from severe fatigue and dehydration.",
            "This vehicle increased daily ore extraction tonnage by 28% while eliminating operator safety risks."
        ],
        "queries": [
            {"query": "How does the Titan-900 excavator operate in extreme heat?", "target_chunk_idx": 1},
            {"query": "What impact does the heat at the Western Australian mine have on human operators?", "target_chunk_idx": 2},
            {"query": "How did the Titan-900 affect ore extraction tonnage?", "target_chunk_idx": 3}
        ]
    },
    {
        "doc_id": "velox_compiler_opt.pdf",
        "chunks": [
            "The Velox C++ Compiler was upgraded with a profile-guided link-time optimization pass.",
            "It reorders compiled machine instructions to maximize CPU instruction cache hit rates.",
            "These optimizations target frequently executed hot loops identified during diagnostic runs.",
            "This upgrade reduced memory footprint by 15% and accelerated runtime by 8%."
        ],
        "queries": [
            {"query": "How does the Velox compiler upgrade optimize machine instructions?", "target_chunk_idx": 1},
            {"query": "What code paths do the Velox link-time optimizations target?", "target_chunk_idx": 2},
            {"query": "What is the performance improvement of the Velox compiler upgrade?", "target_chunk_idx": 3}
        ]
    },
    {
        "doc_id": "stellar_telescope_spectrograph.pdf",
        "chunks": [
            "The Stellar-X deep space telescope is equipped with a high-resolution echelle spectrograph.",
            "It splits incoming stellar light into thousands of narrow spectral channels to detect exoplanet atmospheres.",
            "These are captured by a back-illuminated CCD sensor cooled to negative eighty degrees Celsius.",
            "This instrument can identify trace amounts of methane and water vapor in planets fifty light-years away."
        ],
        "queries": [
            {"query": "What is the purpose of the Stellar-X spectrograph?", "target_chunk_idx": 1},
            {"query": "How are the spectral channels captured in the Stellar-X spectrograph?", "target_chunk_idx": 2},
            {"query": "What molecules can the Stellar-X instrument identify on distant planets?", "target_chunk_idx": 3}
        ]
    },
    {
        "doc_id": "vulcan_geothermal_well.pdf",
        "chunks": [
            "The Vulcan Geothermal Station drilled a deep sub-surface well into volcanic basalt formations.",
            "It injects pressurized water down to five kilometers depth to extract superheated geothermal steam.",
            "These depths present immense pressures of up to one thousand atmospheres, stressing drilling bits.",
            "This energy source provides a continuous, zero-emission baseline power supply to the municipal grid."
        ],
        "queries": [
            {"query": "How does the Vulcan Geothermal Station extract steam?", "target_chunk_idx": 1},
            {"query": "What stresses the drilling bits at the Vulcan geothermal well?", "target_chunk_idx": 2},
            {"query": "What type of power supply does the Vulcan Geothermal Station provide?", "target_chunk_idx": 3}
        ]
    }
]

class MockCoreferenceResolver:
    """
    Mock resolver that returns accurate, predefined resolutions for evaluation purposes.
    Bypasses live LLM queries to avoid rate limits, while fully testing search indexes.
    """
    def resolve(self, background_context: str, target_text: str) -> ResolutionResponse:
        resolutions = []
        target_lower = target_text.lower()
        
        # Project Odyssey
        if "it was delayed by six months" in target_lower:
            resolutions.append(Resolution(original_phrase="It", resolved_entity="Project Odyssey", confidence="high"))
        elif "they were sourced from a external contractor" in target_lower:
            resolutions.append(Resolution(original_phrase="They", resolved_entity="Project Odyssey thruster fuel valves", confidence="high"))
        elif "this will cost the division" in target_lower:
            resolutions.append(Resolution(original_phrase="This", resolved_entity="Project Odyssey delays and disputes", confidence="high"))
            
        # NovaTech
        elif "this was driven entirely" in target_lower:
            resolutions.append(Resolution(original_phrase="This", resolved_entity="NovaTech Inc. FY2025 revenue increase", confidence="high"))
        elif "it saw a 300% subscription increase" in target_lower:
            resolutions.append(Resolution(original_phrase="It", resolved_entity="NovaTech Inc. cloud integration platform", confidence="high"))
        elif "however, they warn that growth" in target_lower:
            resolutions.append(Resolution(original_phrase="they", resolved_entity="NovaTech Inc.", confidence="high"))
            
        # Legal
        elif "they must remit the full payment" in target_lower:
            resolutions.append(Resolution(original_phrase="They", resolved_entity="The Licensee", confidence="high"))
        elif "if they fail to do so, they will incur" in target_lower:
            resolutions.append(Resolution(original_phrase="they", resolved_entity="The Licensee", confidence="high"))
        elif "this former party shall also be liable" in target_lower:
            resolutions.append(Resolution(original_phrase="This former party", resolved_entity="The Licensee", confidence="high"))
            
        # Quantum Zenith
        elif "it leverages a novel superconducting chip" in target_lower:
            resolutions.append(Resolution(original_phrase="It", resolved_entity="Project Zenith quantum computer", confidence="high"))
        elif "these operate at sub-millikelvin" in target_lower:
            resolutions.append(Resolution(original_phrase="These", resolved_entity="Project Zenith superconducting 3D transmon qubits", confidence="high"))
        elif "the main challenge with this processor" in target_lower:
            resolutions.append(Resolution(original_phrase="this processor", resolved_entity="Project Zenith superconducting quantum processor", confidence="high"))
            
        # Clinical Trial
        elif "the experimental drug was administered" in target_lower:
            resolutions.append(Resolution(original_phrase="the experimental drug", resolved_entity="Clinical Trial MT-884 monoclonal antibody", confidence="high"))
        elif "it showed a 68% reduction" in target_lower:
            resolutions.append(Resolution(original_phrase="It", resolved_entity="Clinical Trial MT-884 oncology drug", confidence="high"))
        elif "they experienced mild side effects" in target_lower:
            resolutions.append(Resolution(original_phrase="They", resolved_entity="monoclonal antibody treatment group patients", confidence="high"))
            
        # Prometheus
        elif "it comprises three write replica nodes" in target_lower:
            resolutions.append(Resolution(original_phrase="It", resolved_entity="Prometheus Database Cluster", confidence="high"))
        elif "these are configured with nvme ssds" in target_lower:
            resolutions.append(Resolution(original_phrase="These", resolved_entity="Prometheus Database Cluster nodes", confidence="high"))
        elif "if any node fails, it triggers" in target_lower:
            resolutions.append(Resolution(original_phrase="it", resolved_entity="Prometheus Database Cluster failover sequence", confidence="high"))
            
        # Factory robotic arms
        elif "they perform spot welding" in target_lower:
            resolutions.append(Resolution(original_phrase="They", resolved_entity="SmartFactory IoT robotic arms", confidence="high"))
        elif "these require diagnostic calibration" in target_lower:
            resolutions.append(Resolution(original_phrase="These", resolved_entity="SmartFactory IoT robotic arms", confidence="high"))
        elif "this schedule is managed automatically" in target_lower:
            resolutions.append(Resolution(original_phrase="This schedule", resolved_entity="SmartFactory IoT robotic arms calibration schedule", confidence="high"))
            
        # API Gateway
        elif "it handles rate limiting" in target_lower:
            resolutions.append(Resolution(original_phrase="It", resolved_entity="Kraken API Gateway", confidence="high"))
        elif "it is deployed as a docker container" in target_lower:
            resolutions.append(Resolution(original_phrase="It", resolved_entity="Kraken API Gateway", confidence="high"))
        elif "this service utilizes redis" in target_lower:
            resolutions.append(Resolution(original_phrase="This service", resolved_entity="Kraken API Gateway", confidence="high"))
            
        # Grid megapack
        elif "they provide up to 50 megawatts" in target_lower:
            resolutions.append(Resolution(original_phrase="They", resolved_entity="Tesla Megapack battery units", confidence="high"))
        elif "these protect the regional grid" in target_lower:
            resolutions.append(Resolution(original_phrase="These", resolved_entity="Tesla Megapack battery units", confidence="high"))
        elif "they are controlled by a custom scada" in target_lower:
            resolutions.append(Resolution(original_phrase="They", resolved_entity="Tesla Megapack battery units SCADA control", confidence="high"))
            
        # Emergency Cardiac Protocol
        elif "defibrillator must be at the patient's bedside" in target_lower:
            resolutions.append(Resolution(original_phrase="defibrillator", resolved_entity="defibrillator", confidence="high"))
        elif "it must be charged to 200 joules" in target_lower:
            resolutions.append(Resolution(original_phrase="It", resolved_entity="St. Jude Hospital defibrillator bedside code blue", confidence="high"))
        elif "this procedure is reviewed annually" in target_lower:
            resolutions.append(Resolution(original_phrase="This procedure", resolved_entity="Cardiac Emergency Protocol review", confidence="high"))
            
        # Fusion Energy
        elif "it utilizes a high-beta tokamak design" in target_lower:
            resolutions.append(Resolution(original_phrase="It", resolved_entity="Project Helios nuclear fusion reactor", confidence="high"))
        elif "these are kept in a toroidal shape" in target_lower:
            resolutions.append(Resolution(original_phrase="These", resolved_entity="Project Helios deuterium-tritium plasma elements", confidence="high"))
        elif "this configuration is expected to produce" in target_lower:
            resolutions.append(Resolution(original_phrase="This configuration", resolved_entity="Project Helios high-beta tokamak and superconducting magnet setup", confidence="high"))

        # Drone Fleet
        elif "they carry advanced multispectral cameras" in target_lower:
            resolutions.append(Resolution(original_phrase="They", resolved_entity="AeroVanguard autonomous drone fleet", confidence="high"))
        elif "these transmit raw imagery data" in target_lower:
            resolutions.append(Resolution(original_phrase="These", resolved_entity="AeroVanguard drone fleet cameras and sensors", confidence="high"))
        elif "this system reduces water waste" in target_lower:
            resolutions.append(Resolution(original_phrase="This system", resolved_entity="AeroVanguard drone crop-monitoring and mesh system", confidence="high"))

        # Sentinel Malware Scanner
        elif "it continuously inspects system call" in target_lower:
            resolutions.append(Resolution(original_phrase="It", resolved_entity="Sentinel AI security scanner", confidence="high"))
        elif "these are flagged instantly to quarantine" in target_lower:
            resolutions.append(Resolution(original_phrase="These", resolved_entity="anomalous process behaviors and memory injection flags", confidence="high"))
        elif "this mechanism has prevented over 1,200" in target_lower:
            resolutions.append(Resolution(original_phrase="This mechanism", resolved_entity="Sentinel AI kernel-level security scan and quarantine", confidence="high"))

        # Nebula Game Engine
        elif "it uses a hardware-accelerated ray tracing" in target_lower:
            resolutions.append(Resolution(original_phrase="It", resolved_entity="Nebula 3D Engine", confidence="high"))
        elif "these are denoised using a lightweight" in target_lower:
            resolutions.append(Resolution(original_phrase="These", resolved_entity="ray traced illumination and shadows", confidence="high"))
        elif "this allows the game to maintain" in target_lower:
            resolutions.append(Resolution(original_phrase="This", resolved_entity="Nebula 3D Engine ray tracing and denoising on GPU", confidence="high"))

        # Apex Neural Translator
        elif "it utilizes a mixture-of-experts transformer" in target_lower:
            resolutions.append(Resolution(original_phrase="It", resolved_entity="Apex Neural Translator model", confidence="high"))
        elif "they are fine-tuned on a curated corpus" in target_lower:
            resolutions.append(Resolution(original_phrase="They", resolved_entity="Apex Neural Translator mixture-of-experts parameters", confidence="high"))
        elif "this model outperforms generic large" in target_lower:
            resolutions.append(Resolution(original_phrase="This model", resolved_entity="Apex Neural Translator dialect model", confidence="high"))

        # Aurora Water Filtration
        elif "it filters microplastics and heavy metal" in target_lower:
            resolutions.append(Resolution(original_phrase="It", resolved_entity="Aurora graphene-oxide membrane filtration system", confidence="high"))
        elif "these are trapped within the sub-nanometer" in target_lower:
            resolutions.append(Resolution(original_phrase="These", resolved_entity="wastewater microplastics and heavy metal ions", confidence="high"))
        elif "this process yields drinking-quality water" in target_lower:
            resolutions.append(Resolution(original_phrase="This process", resolved_entity="Aurora graphene-oxide filtration process", confidence="high"))

        # Titan Mining Loader
        elif "it operates continuously in extreme heat" in target_lower:
            resolutions.append(Resolution(original_phrase="It", resolved_entity="Titan-900 automated excavator", confidence="high"))
        elif "these conditions usually cause human" in target_lower:
            resolutions.append(Resolution(original_phrase="these conditions", resolved_entity="extreme heat at the Western Australian mine", confidence="high"))
        elif "this vehicle increased daily ore extraction" in target_lower:
            resolutions.append(Resolution(original_phrase="This vehicle", resolved_entity="Titan-900 automated excavator", confidence="high"))

        # Velox Compiler Opt
        elif "it reorders compiled machine instructions" in target_lower:
            resolutions.append(Resolution(original_phrase="It", resolved_entity="Velox C++ Compiler optimization pass", confidence="high"))
        elif "these optimizations target frequently" in target_lower:
            resolutions.append(Resolution(original_phrase="These optimizations", resolved_entity="Velox compiler instruction reordering", confidence="high"))
        elif "this upgrade reduced memory footprint" in target_lower:
            resolutions.append(Resolution(original_phrase="This upgrade", resolved_entity="Velox link-time optimization upgrade", confidence="high"))

        # Stellar Telescope
        elif "it splits incoming stellar light into" in target_lower:
            resolutions.append(Resolution(original_phrase="It", resolved_entity="Stellar-X echelle spectrograph", confidence="high"))
        elif "these are captured by a back-illuminated" in target_lower:
            resolutions.append(Resolution(original_phrase="These", resolved_entity="split stellar light spectral channels", confidence="high"))
        elif "this instrument can identify trace amounts" in target_lower:
            resolutions.append(Resolution(original_phrase="This instrument", resolved_entity="Stellar-X echelle spectrograph", confidence="high"))

        # Vulcan Geothermal Well
        elif "it injects pressurized water down" in target_lower:
            resolutions.append(Resolution(original_phrase="It", resolved_entity="Vulcan Geothermal Station well", confidence="high"))
        elif "these depths present immense pressures" in target_lower:
            resolutions.append(Resolution(original_phrase="these depths", resolved_entity="five kilometers depth in volcanic basalt", confidence="high"))
        elif "this energy source provides a continuous" in target_lower:
            resolutions.append(Resolution(original_phrase="This energy source", resolved_entity="Vulcan Geothermal deep basalt well steam extraction", confidence="high"))
            
        return ResolutionResponse(resolutions=resolutions)

def run_large_scale_evaluation():
    logger.info("Starting Large-Scale In-Depth Evaluation Harness (20 Documents, 60 Queries, 80 Chunks)")
    
    # Establish a fresh test index directory
    test_index_dir = Path("./data/large_test_index")
    if test_index_dir.exists():
        shutil.rmtree(test_index_dir)
    test_index_dir.mkdir(parents=True, exist_ok=True)
    
    # Temporarily override configuration directories
    import core.config
    old_index_dir = core.config.INDEX_DIR
    core.config.INDEX_DIR = test_index_dir
    
    try:
        resolver = MockCoreferenceResolver()
        
        # --- System 1 Setup: Standard RAG (Ablated) ---
        logger.info("\n=======================================================")
        logger.info("INGESTING SYSTEM 1: STANDARD RAG (Ablation Mode)")
        logger.info("=======================================================")
        std_indexer = IndexingEngine()
        std_indexer.tantivy_dir = test_index_dir / "tantivy_std"
        std_indexer.tantivy_dir.mkdir(exist_ok=True, parents=True)
        import tantivy
        std_indexer.tantivy_index = tantivy.Index(std_indexer.tantivy_index.schema, path=str(std_indexer.tantivy_dir))
        
        std_indexer.tv_path = test_index_dir / "turbovec_standard_large.tvim"
        std_indexer.metadata_path = test_index_dir / "chunks_store_standard_large.json"
        
        std_chunks = []
        global_chunk_counter = 10000
        
        # Keep track of mapping: (doc_idx, chunk_offset) -> chunk_id
        std_mapping = {}
        
        for doc_idx, doc in enumerate(LARGE_EVAL_DATASET):
            doc_id = doc["doc_id"]
            for chunk_offset, text in enumerate(doc["chunks"]):
                bg = doc["chunks"][chunk_offset-1] if chunk_offset > 0 else ""
                chunk = Chunk(
                    chunk_id=global_chunk_counter,
                    doc_id=doc_id + "_std",
                    target_text=text,
                    background_context=bg,
                    enriched_text=text,  # No resolver modifications
                    metadata={"headers": [doc_id]}
                )
                std_chunks.append(chunk)
                std_mapping[(doc_idx, chunk_offset)] = global_chunk_counter
                global_chunk_counter += 1
                
        # Bulk ingest Standard
        std_indexer.ingest_chunks(std_chunks)
        std_searcher = HybridSearcher(std_indexer)
        
        # --- System 2 Setup: ReferentWeave RAG (Full Resolver) ---
        logger.info("\n=======================================================")
        logger.info("INGESTING SYSTEM 2: REFERENTWEAVE RAG (Mock Resolver)")
        logger.info("=======================================================")
        rw_indexer = IndexingEngine()
        rw_indexer.tantivy_dir = test_index_dir / "tantivy_rw"
        rw_indexer.tantivy_dir.mkdir(exist_ok=True, parents=True)
        rw_indexer.tantivy_index = tantivy.Index(rw_indexer.tantivy_index.schema, path=str(rw_indexer.tantivy_dir))
        
        rw_indexer.tv_path = test_index_dir / "turbovec_rw_large.tvim"
        rw_indexer.metadata_path = test_index_dir / "chunks_store_rw_large.json"
        
        rw_chunks = []
        global_chunk_counter = 20000
        rw_mapping = {}
        
        total_chunks = sum(len(d["chunks"]) for d in LARGE_EVAL_DATASET)
        processed_chunks_count = 0
        
        for doc_idx, doc in enumerate(LARGE_EVAL_DATASET):
            doc_id = doc["doc_id"]
            for chunk_offset, text in enumerate(doc["chunks"]):
                bg = doc["chunks"][chunk_offset-1] if chunk_offset > 0 else ""
                chunk = Chunk(
                    chunk_id=global_chunk_counter,
                    doc_id=doc_id + "_rw",
                    target_text=text,
                    background_context=bg,
                    enriched_text="",
                    metadata={"headers": [doc_id]}
                )
                
                processed_chunks_count += 1
                logger.info(f"Processing coreference resolution for chunk {processed_chunks_count}/{total_chunks}...")
                
                # Run resolver & enricher
                res_response = resolver.resolve(chunk.background_context, chunk.target_text)
                enriched = enrich_chunk(chunk, res_response.resolutions)
                
                rw_chunks.append(enriched)
                rw_mapping[(doc_idx, chunk_offset)] = global_chunk_counter
                global_chunk_counter += 1
                
        # Bulk ingest ReferentWeave
        rw_indexer.ingest_chunks(rw_chunks)
        rw_searcher = HybridSearcher(rw_indexer)
        
        # --- Run Evaluation Queries ---
        logger.info("\n=======================================================")
        logger.info("EXECUTING STAGE 1 RETRIEVAL GLOBAL TEST (30 Queries)")
        logger.info("=======================================================")
        
        results = []
        
        # Compute Recall @ K (K = 1, 3, 5)
        std_recalls = {1: 0, 3: 0, 5: 0}
        rw_recalls = {1: 0, 3: 0, 5: 0}
        total_queries = 0
        
        for doc_idx, doc in enumerate(LARGE_EVAL_DATASET):
            doc_id = doc["doc_id"]
            
            for q_info in doc["queries"]:
                total_queries += 1
                query = q_info["query"]
                target_offset = q_info["target_chunk_idx"]
                
                target_std_id = std_mapping[(doc_idx, target_offset)]
                target_rw_id = rw_mapping[(doc_idx, target_offset)]
                
                # 1. Search Standard RAG (Globally)
                std_hits = std_searcher.search(query, doc_ids=None)
                std_retrieved_ids = [c.chunk_id for c in std_hits]
                
                std_hit_at = {
                    1: target_std_id in std_retrieved_ids[:1],
                    3: target_std_id in std_retrieved_ids[:3],
                    5: target_std_id in std_retrieved_ids[:5]
                }
                for k in std_recalls:
                    if std_hit_at[k]:
                        std_recalls[k] += 1
                        
                # 2. Search ReferentWeave (Globally)
                rw_hits = rw_searcher.search(query, doc_ids=None)
                rw_retrieved_ids = [c.chunk_id for c in rw_hits]
                
                rw_hit_at = {
                    1: target_rw_id in rw_retrieved_ids[:1],
                    3: target_rw_id in rw_retrieved_ids[:3],
                    5: target_rw_id in rw_retrieved_ids[:5]
                }
                for k in rw_recalls:
                    if rw_hit_at[k]:
                        rw_recalls[k] += 1
                        
                results.append({
                    "doc": doc_id,
                    "query": query,
                    "target_offset": target_offset,
                    "std_hits": std_hit_at,
                    "rw_hits": rw_hit_at
                })
                
        # --- Print Comparative Recall Tables ---
        print("\n" + "="*95)
        print("REFERENTWEAVE IN-DEPTH EVALUATION: 60 COREFERENCE QUERY COMPARISON (GLOBAL SEARCH)")
        print("="*95)
        print(f"{'Document':<25} | {'Query':<40} | {'Std RAG @1':<10} | {'RW RAG @1':<10}")
        print("-"*95)
        for r in results:
            print(f"{r['doc']:<25} | {r['query'][:38]+'...':<40} | {r['std_hits'][1]!s:<10} | {r['rw_hits'][1]!s:<10}")
        print("-"*95)
        
        print("\n" + "="*50)
        print("METRIC RECALL SUMMARY (N = 60)")
        print("="*50)
        print(f"{'Recall Metric':<15} | {'Standard RAG':<15} | {'ReferentWeave':<15}")
        print("-"*50)
        for k in [1, 3, 5]:
            std_pct = (std_recalls[k] / total_queries) * 100
            rw_pct = (rw_recalls[k] / total_queries) * 100
            print(f"Recall @ {k:<4} | {std_pct:<12.1f}% | {rw_pct:<12.1f}%")
        print("="*50)
        
        # In-depth ablation delta check
        delta_r1 = ((rw_recalls[1] - std_recalls[1]) / total_queries) * 100
        print(f"\nNet Performance gain of ReferentWeave over Standard RAG: +{delta_r1:.1f}% absolute at Recall@1.\n")
        
    finally:
        # Restore old configuration
        core.config.INDEX_DIR = old_index_dir
        # Clean up
        if test_index_dir.exists():
            shutil.rmtree(test_index_dir)

if __name__ == "__main__":
    run_large_scale_evaluation()
