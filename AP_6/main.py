import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found. Using a synthetic test object.")
    return image


def detect_edges(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(gray, 50, 150)
    # Build a clean silhouette mask to ignore internal patterns.
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate(
        [thresh[0, :], thresh[-1, :], thresh[:, 0], thresh[:, -1]]
    )
    if np.mean(border) > 127:
        thresh = 255 - thresh

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return gray, edges, mask


def get_largest_contour(binary_image):
    contours_result = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if len(contours_result) == 3:
        _, contours, _ = contours_result
    else:
        contours, _ = contours_result
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def extract_profile_from_contour(contour):
    x, y, w, h = cv2.boundingRect(contour)
    axis_x = x + w / 2.0

    r_by_y = {}
    for point in contour:
        px, py = point[0]
        radius = abs(px - axis_x)
        if py not in r_by_y or radius > r_by_y[py]:
            r_by_y[py] = radius

    y_vals = np.array(sorted(r_by_y.keys()), dtype=np.float64)
    r_vals = np.array([r_by_y[int(yy)] for yy in y_vals], dtype=np.float64)

    return axis_x, y_vals, r_vals


def fit_profile_curve(y_vals, r_vals, degree=5, num_points=300):
    y_min, y_max = y_vals.min(), y_vals.max()
    y_scaled = (y_vals - y_min) / (y_max - y_min)

    coeffs = np.polyfit(y_scaled, r_vals, degree)
    poly = np.poly1d(coeffs)

    y_fit = np.linspace(y_min, y_max, num_points)
    y_fit_scaled = (y_fit - y_min) / (y_max - y_min)
    r_fit = poly(y_fit_scaled)
    r_fit = np.clip(r_fit, 0, None)

    return y_fit, r_fit


def compute_volume_disk_method(y_fit, r_fit):
    # Volume = pi * integral(r(y)^2 dy)
    return np.pi * np.trapezoid(r_fit ** 2, y_fit)


def build_surface_mesh(y_fit, r_fit, num_theta=80):
    theta = np.linspace(0, 2 * np.pi, num_theta)
    y_grid, theta_grid = np.meshgrid(y_fit, theta)
    r_grid = np.tile(r_fit, (num_theta, 1))

    x_grid = r_grid * np.cos(theta_grid)
    z_grid = r_grid * np.sin(theta_grid)

    return x_grid, y_grid, z_grid


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "image2.png"
    image_bgr = load_image(image_path)

    gray, edges, mask = detect_edges(image_bgr)
    contour = get_largest_contour(mask)
    if contour is None:
        print("No contour found. Try a clearer image.")
        return

    axis_x, y_vals, r_vals = extract_profile_from_contour(contour)
    y_fit, r_fit = fit_profile_curve(y_vals, r_vals)
    volume = compute_volume_disk_method(y_fit, r_fit)

    x_mesh, y_mesh, z_mesh = build_surface_mesh(y_fit, r_fit)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(12, 9))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(image_rgb)
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(edges, cmap="gray")
    ax2.set_title("Detected Edges (Canny)")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(mask, cmap="gray")
    ax3.plot(axis_x + r_vals, y_vals, "r.", markersize=2)
    ax3.plot(axis_x + r_fit, y_fit, "b-", linewidth=1)
    ax3.axvline(axis_x, color="yellow", linewidth=1)
    ax3.set_title("Extracted Profile (Right Side)")
    # ax3.invert_yaxis()
    ax3.axis("off")

    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    ax4.plot_surface(x_mesh, y_mesh, z_mesh, color="lightblue", linewidth=0, alpha=0.9)
    ax4.set_title("Reconstructed Surface of Revolution")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y (axis)")
    ax4.set_zlabel("Z")
    max_r = float(np.max(r_fit))
    y_min = float(np.min(y_fit))
    y_max = float(np.max(y_fit))
    ax4.set_xlim(-max_r, max_r)
    ax4.set_ylim(y_min, y_max)
    ax4.set_zlim(-max_r, max_r)
    ax4.set_box_aspect([2 * max_r, y_max - y_min, 2 * max_r])

    plt.tight_layout()
    plt.show()

    print(f"Estimated volume (pixel^3): {volume:.2f}")


if __name__ == "__main__":
    main()
