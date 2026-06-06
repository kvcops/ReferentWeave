import sys
import os
import logging
import pickle
import numpy as np

logger = logging.getLogger("ReferentWeave.turbovec")

def get_lloyd_max_centroids(bit_width: int = 4) -> np.ndarray:
    """
    Computes optimal Lloyd-Max quantizer centroids for a standard normal distribution N(0, 1).
    Uses 1D K-means/Lloyd's iterations for fast convergence.
    """
    np.random.seed(42)  # Stable centroid generation
    num_centroids = 1 << bit_width
    samples = np.random.randn(20000)
    
    # Initialize centroids using uniform quantiles
    percentiles = np.linspace(100.0 / (2 * num_centroids), 100.0 - 100.0 / (2 * num_centroids), num_centroids)
    centroids = np.percentile(samples, percentiles)
    
    # Lloyd-Max iteration loop
    for _ in range(12):
        # Assign samples to closest centroid
        diffs = np.abs(samples[:, np.newaxis] - centroids)
        assignments = np.argmin(diffs, axis=1)
        # Recalculate centroids
        for i in range(num_centroids):
            mask = (assignments == i)
            if np.any(mask):
                centroids[i] = np.mean(samples[mask])
                
    return np.sort(centroids)

# Resolve circular import: attempt to load global turbovec from site-packages by bypassing local path
RealIdMapIndex = None
original_path = sys.path.copy()
original_module = sys.modules.pop("turbovec", None)
try:
    local_parent = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    # Exclude the workspace root, current directory indicators, and empty paths
    sys.path = [p for p in sys.path if p and p != '' and p != '.' and os.path.abspath(p) != local_parent]
    
    import importlib
    global_turbovec = importlib.import_module("turbovec")
    
    RealIdMapIndex = getattr(global_turbovec, "IdMapIndex")
    logger.info("Successfully loaded native turbovec bindings from site-packages.")
except Exception as e:
    logger.warning(f"Native turbovec binary bindings not found or failed to load: {e}. Falling back to pure Python/NumPy implementation.")
finally:
    # Restore original path and module
    sys.path = original_path
    if original_module:
        sys.modules["turbovec"] = original_module

if RealIdMapIndex is not None:
    # Native bindings are active
    IdMapIndex = RealIdMapIndex
else:
    # Fallback to pure Python/NumPy TurboQuant implementation
    class IdMapIndex:
        def __init__(self, dim: int, bit_width: int = 4):
            self.dim = dim
            self.bit_width = bit_width
            
            # 1. Precompute standard normal Lloyd-Max centroids
            self.centroids = get_lloyd_max_centroids(bit_width)
            # Centroids scaled by 1/sqrt(d) since coordinates follow N(0, 1/d) after rotation
            self.centroids_scaled = self.centroids / np.sqrt(self.dim)
            
            # 2. Generate random orthogonal rotation matrix W via QR decomposition
            np.random.seed(42)  # Seed for deterministic testing
            H = np.random.randn(self.dim, self.dim)
            Q, R = np.linalg.qr(H)
            self.W = Q.astype(np.float32)  # Orthogonal rotation matrix
            
            # 3. Storage containers
            self.ids = []                     # List of uint64 ids
            self.id_to_idx = {}               # map id -> row index in matrix
            self.norms = np.zeros(0, dtype=np.float32)  # stored norms ||v||_2
            self.scales = np.zeros(0, dtype=np.float32) # length-renormalization correction scalars (S)
            self.packed_codes = np.zeros((0, self.dim // 2), dtype=np.uint8) # 4-bit packed codes
            
            logger.info(f"Initialized fallback TurboQuant IdMapIndex: dim={dim}, bit_width={bit_width}")
            
        def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray):
            """
            Ingests vectors, normalizes, rotates, quantizes with Lloyd-Max,
            and calculates length-renormalization scale corrections.
            """
            if len(vectors) == 0:
                return
            
            # 1. Normalize: strip norm
            norms = np.linalg.norm(vectors, axis=1)
            norms_safe = np.where(norms == 0, 1e-9, norms)
            unit_vectors = vectors / norms_safe[:, np.newaxis]
            
            # 2. Random rotation: x = W * u
            # shape of unit_vectors: (N, dim). shape of W: (dim, dim)
            rotated = np.dot(unit_vectors, self.W.T)  # (N, dim)
            
            # 3. Scale rotated coordinates to standard normal: y = x * sqrt(d)
            scaled_rotated = rotated * np.sqrt(self.dim)
            
            # 4. Quantize to 16 centroids (4-bit)
            # Find closest centroid for each coordinate in scaled_rotated
            diffs = np.abs(scaled_rotated[:, :, np.newaxis] - self.centroids[np.newaxis, np.newaxis, :])
            codes = np.argmin(diffs, axis=2).astype(np.uint8)  # shape (N, dim)
            
            # 5. Length-Renormalization scale correction: S = ||v||_2 / <x, x'>
            # Reconstruct rotated unit vector coordinate values from centroid index
            reconstructed_rotated = self.centroids_scaled[codes]  # shape (N, dim)
            
            # Compute inner products of original rotated and reconstructed rotated
            inner_products = np.sum(rotated * reconstructed_rotated, axis=1)
            inner_products_safe = np.where(inner_products == 0, 1e-9, inner_products)
            scales = norms / inner_products_safe
            
            # 6. Bit pack: combine two 4-bit values into one 8-bit byte
            codes_even = codes[:, ::2]
            codes_odd = codes[:, 1::2]
            packed = (codes_even << 4) | codes_odd  # shape (N, dim // 2)
            
            # 7. Merge into storage matrices
            for idx, vec_id in enumerate(ids):
                vid = int(vec_id)
                if vid in self.id_to_idx:
                    # Update in place
                    row = self.id_to_idx[vid]
                    self.norms[row] = norms[idx]
                    self.scales[row] = scales[idx]
                    self.packed_codes[row] = packed[idx]
                else:
                    # Append new
                    row = len(self.ids)
                    self.ids.append(vid)
                    self.id_to_idx[vid] = row
                    
                    self.norms = np.append(self.norms, norms[idx])
                    self.scales = np.append(self.scales, scales[idx])
                    if self.packed_codes.shape[0] == 0:
                        self.packed_codes = packed[idx:idx+1]
                    else:
                        self.packed_codes = np.vstack([self.packed_codes, packed[idx]])
                        
            logger.debug(f"Ingested {len(ids)} vectors into fallback TurboQuant index.")
            
        def remove(self, id_val: int):
            """Removes a vector from the index in O(N)."""
            vid = int(id_val)
            if vid not in self.id_to_idx:
                return
                
            row_to_remove = self.id_to_idx[vid]
            
            # Delete from list and arrays
            self.ids.pop(row_to_remove)
            self.norms = np.delete(self.norms, row_to_remove)
            self.scales = np.delete(self.scales, row_to_remove)
            self.packed_codes = np.delete(self.packed_codes, row_to_remove, axis=0)
            
            # Rebuild index map
            self.id_to_idx = {vid: idx for idx, vid in enumerate(self.ids)}
            logger.debug(f"Removed vector ID {id_val} from fallback TurboQuant index.")
            
        def search(self, query_vector: np.ndarray, k: int, allowlist: np.ndarray = None):
            """
            Performs length-renormalized cosine similarity search over 4-bit packed codes.
            """
            if len(self.ids) == 0:
                return np.array([], dtype=np.float32), np.array([], dtype=np.uint64)
                
            # Filter by allowlist
            if allowlist is not None and len(allowlist) > 0:
                allowed_set = set(int(x) for x in allowlist)
                active_rows = [self.id_to_idx[vid] for vid in self.ids if vid in allowed_set]
            else:
                active_rows = list(range(len(self.ids)))
                
            if not active_rows:
                return np.array([], dtype=np.float32), np.array([], dtype=np.uint64)
                
            # 1. Normalize and rotate query
            q_norm = np.linalg.norm(query_vector)
            q_norm = 1e-9 if q_norm == 0 else q_norm
            q_unit = query_vector / q_norm
            
            if q_unit.ndim == 2:
                q_unit = q_unit.flatten()
            q_rot = np.dot(self.W, q_unit)  # (dim,)
            
            # 2. Slice active matrices
            active_packed = self.packed_codes[active_rows]  # (M, dim // 2)
            active_scales = self.scales[active_rows]        # (M,)
            active_ids = np.array([self.ids[r] for r in active_rows], dtype=np.uint64)
            
            # 3. Unpack codes
            codes_even = active_packed >> 4
            codes_odd = active_packed & 0x0F
            
            unpacked = np.empty((active_packed.shape[0], self.dim), dtype=np.uint8)
            unpacked[:, ::2] = codes_even
            unpacked[:, 1::2] = codes_odd  # (M, dim)
            
            # 4. Reconstruct rotated unit vectors from Lloyd-Max centroids
            reconstructed_rotated = self.centroids_scaled[unpacked]  # (M, dim)
            
            # 5. Score computation: score_uncalibrated = <u_rot, q_rot>
            scores = np.dot(reconstructed_rotated, q_rot)  # (M,)
            
            # 6. Apply length-renormalized correction scalar (S)
            calibrated_scores = scores * active_scales  # (M,)
            
            # 7. Get top K
            k = min(k, len(calibrated_scores))
            top_k_indices = np.argsort(calibrated_scores)[::-1][:k]
            
            return calibrated_scores[top_k_indices].astype(np.float32), active_ids[top_k_indices].astype(np.uint64)
            
        def write(self, path: str):
            """Serializes the index state including rotation matrix and centroids."""
            with open(path, "wb") as f:
                pickle.dump({
                    "dim": self.dim,
                    "bit_width": self.bit_width,
                    "centroids": self.centroids,
                    "W": self.W,
                    "ids": self.ids,
                    "norms": self.norms,
                    "scales": self.scales,
                    "packed_codes": self.packed_codes
                }, f)
            logger.info(f"Wrote fallback TurboQuant index of size {len(self.ids)} to {path}")
                
        @classmethod
        def load(cls, path: str):
            """Loads index state from disk."""
            with open(path, "rb") as f:
                data = pickle.load(f)
            index = cls(dim=data["dim"], bit_width=data["bit_width"])
            index.centroids = data["centroids"]
            index.centroids_scaled = index.centroids / np.sqrt(index.dim)
            index.W = data["W"]
            index.ids = data["ids"]
            index.id_to_idx = {vid: idx for idx, vid in enumerate(index.ids)}
            index.norms = data["norms"]
            index.scales = data["scales"]
            index.packed_codes = data["packed_codes"]
            logger.info(f"Loaded fallback TurboQuant index of size {len(index.ids)} from {path}")
            return index
