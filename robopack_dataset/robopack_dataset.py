from typing import Iterator, Tuple, Any
import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# === Constants to be adjusted based on your data ===
IMAGE_H = 480
IMAGE_W = 640
TACTILE_H = 128
TACTILE_W = 128

class RobopackDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for the RoboPack dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release with RGB-D, bubble sensors, and robot states.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'images': tfds.features.FeaturesDict({
                            f'cam_{i}': tfds.features.Tensor(shape=(IMAGE_H, IMAGE_W, 3), dtype=tf.uint8)
                            for i in range(4)
                        }),
                        'depths': tfds.features.FeaturesDict({
                            f'cam_{i}': tfds.features.Tensor(shape=(IMAGE_H, IMAGE_W), dtype=tf.float32)
                            for i in range(4)
                        }),
                        'bubble_depths': tfds.features.FeaturesDict({
                            f'bubble_{i}': tfds.features.Tensor(shape=(TACTILE_H, TACTILE_W), dtype=tf.float32)
                            for i in range(2)
                        }),
                        'bubble_forces': tfds.features.FeaturesDict({
                            f'bubble_{i}': tfds.features.Tensor(shape=(TACTILE_H, TACTILE_W), dtype=tf.float32)
                            for i in range(2)
                        }),
                        'bubble_flows': tfds.features.FeaturesDict({
                            f'bubble_{i}': tfds.features.Tensor(shape=(TACTILE_H, TACTILE_W, 2), dtype=tf.float32)
                            for i in range(2)
                        }),
                        'ee_pose': tfds.features.Tensor(shape=(7,), dtype=tf.float32),
                        'joint_positions': tfds.features.Tensor(shape=(7,), dtype=tf.float32),
                    }),
                    'action': tfds.features.Tensor(shape=(7,), dtype=tf.float32),  # Placeholder
                    'discount': tfds.features.Scalar(dtype=tf.float32),
                    'reward': tfds.features.Scalar(dtype=tf.float32),
                    'is_first': tfds.features.Scalar(dtype=tf.bool),
                    'is_last': tfds.features.Scalar(dtype=tf.bool),
                    'is_terminal': tfds.features.Scalar(dtype=tf.bool),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'seq_id': tfds.features.Text(),
                    'path': tfds.features.Text(),
                }),
            }),
            description="RoboPack dataset containing synchronized RGB-D, tactile, and robot state data.",
            supervised_keys=None,
            homepage="https://example.com/robopack",
            citation="TBD",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        root_dir = "/home/albert/Downloads/robopack-data/1_6_box5"
        return {
            'train': self._generate_examples(os.path.join(root_dir, "train")),
            'val': self._generate_examples(os.path.join(root_dir, "val")),
        }

    def _generate_examples(self, split_path) -> Iterator[Tuple[str, Any]]:
        seq_dirs = sorted(glob.glob(os.path.join(split_path, "seq_*")))

        for seq_path in seq_dirs:
            seq_id = os.path.basename(seq_path)

            # Infer all timestamps from ee_states
            ee_files = sorted(glob.glob(os.path.join(seq_path, "ee_states", "ee_states_*.npy")))
            timestamps = [
                "_".join(f.split("_")[-2:]).replace(".npy", "")
                for f in ee_files
            ]

            episode = []
            for i, ts in enumerate(timestamps):
                t_index = ts.split("_")[0]
                cam_index = ts.split("_")[1]

                step = {
                    'observation': {
                        'images': {},
                        'depths': {},
                        'bubble_depths': {},
                        'bubble_forces': {},
                        'bubble_flows': {},
                        'ee_pose': np.load(os.path.join(seq_path, "ee_states", f"ee_states_{ts}.npy")).astype(np.float32),
                        'joint_positions': np.load(os.path.join(seq_path, "joint_states", f"joint_states_{ts}.npy")).astype(np.float32),
                    },
                    'action': np.zeros(7, dtype=np.float32),  # Placeholder
                    'discount': 1.0,
                    'reward': float(i == len(timestamps) - 1),
                    'is_first': i == 0,
                    'is_last': i == len(timestamps) - 1,
                    'is_terminal': i == len(timestamps) - 1,
                }

                for cam_id in range(4):
                    cam_path = os.path.join(seq_path, f"cam_{cam_id}")
                    color_file = os.path.join(cam_path, f"color_{t_index}_{cam_id}.npy")
                    depth_file = os.path.join(cam_path, f"depth_{t_index}_{cam_id}.npy")
                    if os.path.exists(color_file):
                        color = np.load(color_file).clip(0, 255).astype(np.uint8)
                        step['observation']['images'][f'cam_{cam_id}'] = color
                    if os.path.exists(depth_file):
                        depth = np.load(depth_file).astype(np.float32)
                        step['observation']['depths'][f'cam_{cam_id}'] = depth

                for b_id in range(2):
                    b_path = os.path.join(seq_path, f"bubble_{b_id}")
                    for modality in ['depth', 'force', 'raw_flow']:
                        file = os.path.join(b_path, f"{modality}_{t_index}_{b_id}.npy")
                        if os.path.exists(file):
                            data = np.load(file).astype(np.float32)
                            if modality == 'depth':
                                step['observation']['bubble_depths'][f'bubble_{b_id}'] = data
                            elif modality == 'force':
                                step['observation']['bubble_forces'][f'bubble_{b_id}'] = data
                            elif modality == 'raw_flow':
                                step['observation']['bubble_flows'][f'bubble_{b_id}'] = data

                episode.append(step)

            yield seq_id, {
                'steps': episode,
                'episode_metadata': {
                    'seq_id': seq_id,
                    'path': seq_path,
                }
            }
            
