import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.cluster.hierarchy import dendrogram

# =========================================================
# 1. TAG AXES — 6D SEMANTIC SPACE
# =========================================================
tag_vectors = {
    # Math
    "math":            np.array([1,0,0,0,0,0]),
    "linear-algebra":  np.array([0.9,0,0,0,0,0]),
    "calculus":        np.array([0.85,0,0,0,0,0]),
    "analysis":        np.array([0.8,0,0,0,0,0]),

    # Coding
    "coding":          np.array([0,1,0,0,0,0]),
    "python":          np.array([0,0.9,0,0,0,0]),
    "cpp":             np.array([0,0.8,0,0,0,0]),
    "algorithms":      np.array([0,0.85,0,0,0,0]),
    "project":         np.array([0,0.7,0,0,0,0.2]),

    # Art
    "art":             np.array([0,0,1,0,0,0]),
    "drawing":         np.array([0,0,0.9,0,0,0]),
    "design":          np.array([0,0,0.8,0,0,0]),

    # Fantasy / Worldbuilding
    "fantasy":         np.array([0,0,0,1,0.1,0]),
    "worldbuilding":   np.array([0,0,0,0.9,0.05,0]),
    "lore":            np.array([0,0,0,0.8,0.05,0]),
    "dnd":             np.array([0,0,0,0.8,0.2,0]),

    # Gaming
    "gaming":          np.array([0,0,0,0.1,1,0]),
    "helldivers":      np.array([0,0,0,0,0.9,0]),
    "game-dev":        np.array([0,0.5,0,0.1,0.9,0]),

    # Hobby
    "hobby":           np.array([0,0,0,0,0,1]),
    "journal":         np.array([0,0,0,0,0,0.8]),
    "recipes":         np.array([0,0,0,0,0,0.9]),
}

# =========================================================
# 2. STRESS-TEST DOCUMENT DATASET (40+ docs)
# =========================================================
documents = {
    "la_notes": ["math", "linear-algebra"],
    "calc_homework": ["math", "calculus"],
    "analysis_summary": ["math", "analysis"],
    "probability_notes": ["math"],

    "python_basics": ["coding", "python"],
    "cpp_optimizations": ["coding", "cpp"],
    "algo_design": ["coding", "algorithms"],
    "project_todo": ["coding", "project"],

    "portrait_sketch": ["art", "drawing"],
    "ui_design_draft": ["art", "design"],
    "color_theory": ["art"],

    "dnd_session_1": ["dnd", "worldbuilding"],
    "world_map": ["fantasy", "worldbuilding"],
    "ancient_lore": ["fantasy", "lore"],
    "magic_system": ["fantasy"],

    "game_review_hd2": ["gaming", "helldivers"],
    "build_guide_slayer": ["gaming"],
    "game_dev_plan": ["gaming", "game-dev", "project"],

    "daily_journal": ["hobby", "journal"],
    "recipe_tacos": ["hobby", "recipes"],
    "fitness_log": ["hobby"],

    # Mixed
    "math_for_algos": ["math", "algorithms"],
    "python_numerics": ["coding", "math", "python"],
    "machine_learning_intro": ["math", "coding", "algorithms", "python"],
    "project_summary": ["coding", "math", "project"],

    "python_for_dnd": ["coding", "dnd"],
    "campaign_tracker_tool": ["coding", "fantasy", "project"],
    "map_generator_code": ["coding", "worldbuilding", "project"],
    "spell_calc_program": ["coding", "fantasy", "math"],

    "character_concepts": ["art", "fantasy"],
    "world_map_art": ["art", "worldbuilding"],
    "magic_creature_design": ["art", "fantasy", "design"],
    "ui_for_fantasy_game": ["art", "gaming", "design"],

    "ai_bot_for_games": ["coding", "gaming", "algorithms"],
    "game_modding_notes": ["gaming", "coding"],
    "game_dev_diary": ["gaming", "project", "journal"],
    "procedural_generation": ["coding", "game-dev", "fantasy"],

    "math_of_color": ["math", "art"],
    "fantasy_statistics": ["math", "fantasy"],
    "cooking_for_dnd_party": ["hobby", "fantasy", "recipes"],
    "art_of_algorithms": ["math", "coding", "art"],
    "lore_based_ai": ["coding", "fantasy", "algorithms"],
    "game_study_notebook": ["math", "gaming", "journal"],
    "dnd_budget_sheet": ["math", "hobby", "dnd"],
    "fantasy_cooking": ["fantasy", "recipes"],
}

# =========================================================
# 3. DOCUMENT → VECTOR
# =========================================================
def document_vector(tags):
    vecs = [tag_vectors[t] for t in tags if t in tag_vectors]
    return np.mean(vecs, axis=0) if vecs else np.zeros(6)

names = list(documents.keys())
X = np.vstack([document_vector(documents[n]) for n in names])


# =========================================================
# 4. CLUSTER DISTANCE
# =========================================================
def cluster_distance(array, clusterA, clusterB, norm=np.linalg.norm):
    return min(
        norm(array[i] - array[j])
        for i in clusterA
        for j in clusterB
    )

# =========================================================
# 5. HIERARCHICAL CLUSTERING (WITH MERGE HISTORY)
# =========================================================
def clustering_with_history(array):
    N = len(array)
    clusters = [{i} for i in range(N)]
    ids = list(range(N))
    next_id = N
    history = []

    while len(clusters) > 1:
        best = None
        best_dist = float('inf')

        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                d = cluster_distance(array, clusters[i], clusters[j])
                if d < best_dist:
                    best_dist = d
                    best = (i, j)

        i, j = best
        A, B = clusters[i], clusters[j]
        merged = A | B

        history.append((ids[i], ids[j], best_dist, len(merged)))

        # remove and add merged
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)]
        ids = [cid for k, cid in enumerate(ids) if k not in (i, j)]
        clusters.append(merged)
        ids.append(next_id)
        next_id += 1

    return clusters[0], np.array(history, float)


# =========================================================
# 6. RUN CLUSTERING
# =========================================================
root, linkage = clustering_with_history(X)


# =========================================================
# 7. PLOTLY DENDROGRAM
# =========================================================
def plot_dendrogram(linkage, labels):
    d = dendrogram(linkage, labels=labels, orientation="left", no_plot=True)

    fig = go.Figure()

    # ---- Draw dendrogram lines ----
    for xs, ys in zip(d['dcoord'], d['icoord']):
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode='lines',
            line=dict(color='black')
        ))

    # ---- Add labels on the left ----
    leaf_order = d['leaves']      # order of documents in the plot
    leaf_positions = d['icoord']  # cluster Y positions

    # Each leaf appears exactly once; find its Y position
    y_coords = {}
    for xs, ys in zip(d['dcoord'], d['icoord']):
        # leaves always have a zero distance on left side = xs[0] == 0
        if xs[0] == 0:
            # ys is a list of 4 values, leaves appear at ys[1] or ys[2]
            # both correspond to leaf nodes; pick ys[1]
            y_coords[ys[1]] = None

    # SciPy’s d['ivl'] (inverted leaf labels) gives label order top→bottom
    ordered_labels = d['ivl']
    ordered_y = sorted(y_coords.keys(), reverse=False)

    for lbl, y in zip(ordered_labels, ordered_y):
        fig.add_trace(go.Scatter(
            x=[-0.5],  # left side
            y=[y],
            mode="text",
            text=[lbl],
            textposition="middle right",
            showlegend=False
        ))

    fig.update_layout(
        title="Hierarchical Clustering Dendrogram",
        width=1300,
        height=min(2500, 50 * len(labels)),  # automatic scaling
        xaxis_title="Distance",
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
    )

    fig.show()


plot_dendrogram(linkage, names)


