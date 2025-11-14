import numpy as np


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum()))


def cluster_distance(points: np.ndarray, cluster_a: set[int], cluster_b: set[int]) -> float:
    # single-linkage: distance between closest members helps keep implementation short
    return min(
        euclidean_distance(points[i], points[j])
        for i in cluster_a
        for j in cluster_b
    )


def hierarchical_clustering(points: np.ndarray, target_clusters: int) -> list[set[int]]:
    clusters: list[set[int]] = [{idx} for idx in range(len(points))]

    while len(clusters) > target_clusters:
        best_pair = None
        best_dist = float("inf")

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = cluster_distance(points, clusters[i], clusters[j])
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (i, j)

        if best_pair is None:  # defensive; should not happen
            break

        i, j = best_pair
        merged = clusters[i] | clusters[j]
        clusters = [
            cluster for idx, cluster in enumerate(clusters) if idx not in (i, j)
        ]
        clusters.append(merged)

    return clusters


def main() -> None:
    rng = np.random.default_rng(seed=7)
    data = rng.normal(loc=(0, 0), scale=0.5, size=(5, 2))
    data = np.vstack([data, rng.normal(loc=(3, 3), scale=0.5, size=(5, 2))])

    clusters = hierarchical_clustering(data, target_clusters=2)

    print("Points:")
    for idx, pt in enumerate(data):
        print(f"{idx}: {pt}")

    print("\nClusters:")
    for cluster_id, members in enumerate(clusters, start=1):
        coords = [data[idx] for idx in members]
        print(f"{cluster_id}: indices={sorted(members)}, centroid={np.mean(coords, axis=0)}")


if __name__ == "__main__":
    main()
