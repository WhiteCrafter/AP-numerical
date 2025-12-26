# AP_6: 3D Reconstruction from a Symmetric Object

## What the project does
This project loads a picture of a vertically symmetric object (vase, bottle, glass, etc.), detects its edges, extracts the 2D profile, and reconstructs a 3D surface by rotating the profile around the symmetry axis. It also estimates the volume using numerical integration.

## Mathematical idea
If an object is symmetric around a vertical axis, we only need its radius as a function of height. By rotating the profile curve 360 degrees, we get a surface of revolution. The volume can be computed with the disk method:

Volume = pi * integral( r(y)^2 dy )

Here, r(y) is the distance from the axis to the edge at height y.

## Image processing steps
1. Read the image and convert it to grayscale.
2. Apply Gaussian blur to reduce noise.
3. Use Canny edge detection to get the object boundary.
4. Select the largest contour as the object.
5. For each y-value, measure the farthest edge point from the symmetry axis to get the profile.
6. Fit a smooth polynomial to the profile points.

## Numerical method for volume
The code uses the disk method and computes the integral numerically with the trapezoidal rule:

Volume ≈ pi * trapz( r(y)^2, y )

The result is in pixel^3 units because the image is measured in pixels.

## How to run
From the AP_6 folder:

```bash
python main.py path_to_image.jpg
```

If no image is given or the file is missing, the script creates a simple synthetic object so the process can still be demonstrated.

## Conclusion
This project shows how a 2D edge profile can be turned into a 3D shape using symmetry. The combination of edge detection, curve fitting, and numerical integration provides a simple but complete pipeline for estimating volume from an image.
