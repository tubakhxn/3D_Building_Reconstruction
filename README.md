# 3D Building Reconstruction from drone footage

**Drone Photogrammetry | Structure-from-Motion | Multi-View Stereo**

**Developer/Creator: tubakhxn**

Research & Educational Purposes Only.

A 3D reconstruction of a building built from a single 4K drone video. It uses classical computer vision (feature matching, camera pose estimation, dense stereo, surface reconstruction). It does not use object detection or generated imagery, and no geometry is made up.

---

## Results

| Item | Value |
|---|---|
| Input video | 3840x2160, 29.97 fps, 13.5 s, 405 frames |
| Frames used | 51 (sharpest, well-spaced, resized to 1600x900) |
| Verified frame pairs | 236 (about 4,480 matches per pair) |
| Registered cameras | 51 of 51 |
| Sparse point cloud | 20,788 points (mean track length 8.8 views) |
| Bundle adjustment error | 0.86 px RMS |
| Dense point cloud | 117,701 points (after outlier removal) |
| Mesh | 15,087 vertices, 24,386 triangles |
| Output videos | 1280x720, 30 fps, H.264 |

---

## Pipeline

1. **Frame selection** (`step1_frames.py`): picks 51 sharp frames with good overlap from the 405.
2. **Feature matching** (`step2_features.py`): SIFT keypoints (about 9,000 per frame), ratio test, and MAGSAC geometric verification.
3. **Structure-from-Motion** (`step3_sfm.py`): incremental camera pose estimation, triangulation, and sparse bundle adjustment (SciPy).
4. **Multi-View Stereo** (`step4_mvs.py`, `step4b_fuse.py`): plane-sweep depth maps from the estimated poses, multi-view consistency filtering, then voxel and statistical outlier cleaning.
5. **Mesh** (`step5_mesh.py`): TSDF fusion around the main building, then marching cubes. Vertices are colored from the video frames.
6. **Video** (`render_video.py`, `render_lib.py`): renders all scenes, labels and watermark, and encodes with FFmpeg.

`main.py` runs everything in order.

---

## Video scenes

1. Original drone footage
2. Feature detection and matching
3. Camera pose estimation (trajectory and coordinate axes)
4. Structure-from-Motion (sparse point cloud)
5. Multi-View Stereo (dense colored point cloud)
6. 3D reconstructed building (mesh)
7. Final comparison: real footage next to the reconstruction rendered from the estimated camera poses

---

## Project structure

```
3D_Building_Reconstruction/
    input/
        original_drone.mp4
    output/
        3d_building_reconstruction.mp4
        sparse_point_cloud.mp4
        dense_reconstruction.mp4
    reconstruction/
        camera_poses/cameras.json
        point_cloud/sparse_point_cloud.ply
        point_cloud/dense_point_cloud.ply
        mesh/building_mesh.ply
        frames/
    main.py
    step1_frames.py ... step5_mesh.py
    render_video.py
    render_lib.py
    requirements.txt
    README.md
```

---

## How to run

```
pip install -r requirements.txt
python main.py
```

Requires Python 3.12, FFmpeg on the PATH, and the video at `input/original_drone.mp4`. Runs on CPU only. A full run takes roughly 15 to 20 minutes on a single core.

---

## Software versions

Python 3.12.3, OpenCV 5.0.0, NumPy 2.5.4, SciPy 1.18.1, scikit-image 0.26.0, FFmpeg.

COLMAP, OpenMVS and Open3D could not be installed in the build environment (no network access), so the equivalent stages were implemented directly with the libraries above.

---

## Limitations

- **Not metric.** The video has no camera metadata, so the focal length was assumed (about 1050 px at 1600 px width, roughly 75 degrees horizontal field of view). For this forward-moving clip, focal lengths from 700 to 2000 px fit equally well. No real-world dimensions are claimed.
- **Depth smearing.** Forward flight gives little parallax, so geometry stretches along the viewing direction in side views. The camera orbits in the videos stay within about 60 degrees of the original flight direction.
- **Incomplete mesh.** Only surfaces confirmed by several depth maps are reconstructed. Facades seen at shallow angles and trees have holes, and the back of the building is not reconstructed. Gaps are left empty on purpose.
- **No lens-distortion calibration.**
- **No Gaussian Splatting** (no GPU available).

---

## Credits

Developer/Creator: **tubakhxn**
