"""
Face ID clustering based on ArcFace embeddings.
Uses DBSCAN for robust clustering (reference: FacesetProcessor/Filter.py).
"""
import numpy as np
from typing import Dict, List, Optional, Tuple


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    return float(np.dot(a, b))


def cluster_embeddings_dbscan(
    embeddings: Dict[str, np.ndarray],
    eps: float = 0.3,
    min_samples: int = 1
) -> Dict[str, list]:
    """
    Cluster face embeddings using DBSCAN.

    Args:
        embeddings: {face_key: L2-normalized embedding}
        eps: DBSCAN eps in cosine distance space (default 0.3 ≈ cos >= 0.7)
        min_samples: DBSCAN min_samples

    Returns:
        {representative_key: [member_key_1, member_key_2, ...]}
        Noise points are stored under key "__noise__".
    """
    keys = list(embeddings.keys())
    if len(keys) == 0:
        return {}
    if len(keys) == 1:
        return {keys[0]: [keys[0]]}

    from sklearn.cluster import DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity as sk_cos_sim

    matrix = np.array([embeddings[k] for k in keys])
    sim = sk_cos_sim(matrix)
    dist = np.clip(1.0 - sim, 0.0, None)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = db.fit_predict(dist)

    clusters = {}
    noise = []
    for i, label in enumerate(labels):
        if label == -1:
            noise.append(keys[i])
        else:
            clusters.setdefault(label, []).append(keys[i])

    # Renumber: largest cluster first
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)
    result = {}
    for group in sorted_clusters:
        result[group[0]] = group
    if noise:
        result["__noise__"] = noise
    return result


def cluster_embeddings(
    embeddings: Dict[str, np.ndarray],
    threshold: float = 0.95
) -> Dict[str, list]:
    """
    Group face IDs by embedding similarity.
    Legacy variant using greedy single-linkage (kept for backward compat).
    Threshold is cosine similarity (higher = stricter).
    """
    keys = list(embeddings.keys())
    assigned = set()
    clusters = {}

    for i, k1 in enumerate(keys):
        if k1 in assigned:
            continue
        cluster = [k1]
        assigned.add(k1)
        for k2 in keys[i + 1:]:
            if k2 in assigned:
                continue
            sim = cosine_similarity(embeddings[k1], embeddings[k2])
            if sim >= threshold:
                cluster.append(k2)
                assigned.add(k2)
        clusters[cluster[0]] = cluster
    return clusters


def match_face_to_cluster(
    embedding: np.ndarray,
    clusters: Dict[str, list],
    all_embeddings: Dict[str, np.ndarray],
    threshold: float = 0.7
) -> Optional[str]:
    """
    Match a face embedding to the closest cluster.
    Returns the main cluster key, or None if no match above threshold.
    """
    best_id = None
    best_sim = 0.0
    for main_id, members in clusters.items():
        if main_id == "__noise__":
            continue
        for member in members:
            if member not in all_embeddings:
                continue
            sim = cosine_similarity(embedding, all_embeddings[member])
            if sim > best_sim:
                best_sim = sim
                best_id = main_id
    return best_id if best_sim >= threshold else None
