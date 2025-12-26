# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""
Preview node for SAM 3D Body rigged meshes.

Displays rigged FBX files with interactive skeleton manipulation.
"""

import os
import numpy as np
import folder_paths


class SAM3DBodyPreviewRiggedMesh:
    """
    Preview rigged mesh with interactive FBX viewer.

    Displays the rigged FBX in a Three.js viewer with skeleton visualization
    and interactive bone manipulation controls.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fbx_output_path": ("STRING", {
                    "tooltip": "FBX filename from output directory (from SAM3D export node)"
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "SAM3DBody/visualization"

    def preview(self, fbx_output_path):
        """Preview the rigged mesh in an interactive FBX viewer."""
        print(f"[SAM3DBodyPreviewRiggedMesh] Preparing preview...")

        # FBX should already be in output directory
        output_dir = folder_paths.get_output_directory()
        fbx_path = os.path.join(output_dir, fbx_output_path)

        if not os.path.exists(fbx_path):
            raise RuntimeError(f"FBX file not found in output directory: {fbx_output_path}")

        print(f"[SAM3DBodyPreviewRiggedMesh] FBX path: {fbx_path}")

        # FBX is already in output, so viewer can access it directly
        # Assume all FBX files have skinning and skeleton
        has_skinning = True
        has_skeleton = True

        print(f"[SAM3DBodyPreviewRiggedMesh] Has skinning: {has_skinning}")
        print(f"[SAM3DBodyPreviewRiggedMesh] Has skeleton: {has_skeleton}")

        return {
            "ui": {
                "fbx_file": [fbx_output_path],
                "has_skinning": [bool(has_skinning)],
                "has_skeleton": [bool(has_skeleton)],
            }
        }


class SAM3DBodyPreviewSkeleton:
    """
    Preview skeleton (bones only, no mesh) with interactive 3D viewer.

    Displays skeleton joints and bones in a Three.js viewer with
    rotation, zoom, and pan controls.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "skeleton": ("SKELETON", {
                    "tooltip": "Skeleton data from SAM3D Body"
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "SAM3DBody/visualization"

    def preview(self, skeleton):
        """Preview skeleton in interactive 3D viewer."""
        import json
        import numpy as np
        import torch

        print(f"[SAM3DBodyPreviewSkeleton] Preparing skeleton preview...")

        # Get joint positions
        joint_positions = skeleton.get("joint_positions")

        if joint_positions is None:
            raise RuntimeError("Skeleton has no joint_positions data")

        # Convert to numpy if needed
        if isinstance(joint_positions, torch.Tensor):
            joint_positions = joint_positions.cpu().numpy()

        # Convert to list for JSON serialization
        joint_positions_list = joint_positions.tolist()

        # Get joint rotations if available
        joint_rotations = skeleton.get("joint_rotations")
        joint_rotations_list = None

        if joint_rotations is not None:
            if isinstance(joint_rotations, torch.Tensor):
                joint_rotations = joint_rotations.cpu().numpy()
            joint_rotations_list = joint_rotations.tolist()

        # Get joint parents for proper bone hierarchy
        joint_parents = skeleton.get("joint_parents")
        joint_parents_list = None

        if joint_parents is not None:
            if isinstance(joint_parents, np.ndarray):
                joint_parents_list = joint_parents.tolist()
            elif isinstance(joint_parents, torch.Tensor):
                joint_parents_list = joint_parents.cpu().numpy().tolist()
            else:
                joint_parents_list = list(joint_parents)

        # MHR70 skeleton links (65 bone connections for first 70 keypoints)
        # Derived from mhr70.py skeleton_info - maps joint names to indices
        # Format: [[joint_a_idx, joint_b_idx], ...]
        skeleton_links = [
            [13, 11], [11, 9], [14, 12], [12, 10],  # legs
            [9, 10], [5, 9], [6, 10], [5, 6],  # torso
            [5, 7], [6, 8], [7, 62], [8, 41],  # arms (wrist indices: left=62, right=41)
            [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],  # face/head
            [13, 15], [13, 16], [13, 17], [14, 18], [14, 19], [14, 20],  # feet
            # Left hand (wrist=62)
            [62, 45], [45, 44], [44, 43], [43, 42],  # thumb
            [62, 49], [49, 48], [48, 47], [47, 46],  # index
            [62, 53], [53, 52], [52, 51], [51, 50],  # middle
            [62, 57], [57, 56], [56, 55], [55, 54],  # ring
            [62, 61], [61, 60], [60, 59], [59, 58],  # pinky
            # Right hand (wrist=41)
            [41, 24], [24, 23], [23, 22], [22, 21],  # thumb
            [41, 28], [28, 27], [27, 26], [26, 25],  # index
            [41, 32], [32, 31], [31, 30], [30, 29],  # middle
            [41, 36], [36, 35], [35, 34], [34, 33],  # ring
            [41, 40], [40, 39], [39, 38], [38, 37],  # pinky
        ]

        # Save skeleton data to temporary JSON file for viewer
        output_dir = folder_paths.get_output_directory()
        skeleton_json_path = os.path.join(output_dir, "_temp_skeleton_preview.json")

        skeleton_data = {
            "joint_positions": joint_positions_list,
            "joint_rotations": joint_rotations_list,
            "joint_parents": joint_parents_list,
            "skeleton_links": skeleton_links,
            "num_joints": len(joint_positions_list),
        }

        with open(skeleton_json_path, 'w') as f:
            json.dump(skeleton_data, f)

        print(f"[SAM3DBodyPreviewSkeleton] Prepared skeleton with {len(joint_positions_list)} joints")

        return {
            "ui": {
                "skeleton_file": ["_temp_skeleton_preview.json"],
                "num_joints": [len(joint_positions_list)],
            }
        }


# Register nodes
NODE_CLASS_MAPPINGS = {
    "SAM3DBodyPreviewRiggedMesh": SAM3DBodyPreviewRiggedMesh,
    "SAM3DBodyPreviewSkeleton": SAM3DBodyPreviewSkeleton,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBodyPreviewRiggedMesh": "SAM 3D Body: Preview Rigged Mesh",
    "SAM3DBodyPreviewSkeleton": "SAM 3D Body: Preview Skeleton",
}
