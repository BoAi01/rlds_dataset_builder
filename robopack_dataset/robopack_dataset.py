from typing import Iterator, Tuple, Any
import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

IMAGE_H, IMAGE_W = 480, 640         # camera image & depth resolution
FLOW_H, FLOW_W = 320, 480           # reshaped tactile flow
EE_DIM = 7                          # end-effector pose
JOINT_DIM = 8                       # joint state
FORCE_DIM = 7                       # tactile force

class RobopackDataset(tfds.core.GeneratorBasedBuilder):
    """RoboPack dataset: RGB-D, tactile, and robot state data."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {'1.0.0': 'Initial release with corrected data loading and shapes.'}

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="RoboPack: RGB-D, tactile, and robot state dataset for manipulation tasks.",
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
                        'bubble_force': tfds.features.Tensor(shape=(FORCE_DIM,), dtype=tf.float32),
                        'bubble_depth': tfds.features.Tensor(shape=(IMAGE_H, IMAGE_W), dtype=tf.float32),
                        'bubble_flow': tfds.features.Tensor(shape=(FLOW_H, FLOW_W), dtype=tf.float32),
                        'ee_pose': tfds.features.Tensor(shape=(EE_DIM,), dtype=tf.float32),
                        'joint_positions': tfds.features.Tensor(shape=(JOINT_DIM,), dtype=tf.float32),
                    }),
                    'action': tfds.features.Tensor(shape=(JOINT_DIM,), dtype=tf.float32),
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
            supervised_keys=None,
            homepage="https://robo-pack.github.io/",
            citation="""
            @article{ai2024robopack,
            title={RoboPack: Learning Tactile-Informed Dynamics Models for Dense Packing},
            author={Bo Ai and Stephen Tian and Haochen Shi and Yixuan Wang and Cheston Tan and Yunzhu Li and Jiajun Wu},
            journal={Robotics: Science and Systems (RSS)},
            year={2024},
            url={https://arxiv.org/abs/2407.01418},
            }
            """,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        root_dir = "/home/albert/Downloads/robopack-data/1_6_box5"
        return {
            'train': self._generate_examples(os.path.join(root_dir, "train")),
            'val': self._generate_examples(os.path.join(root_dir, "val")),
        }

    def _generate_examples(self, split_path) -> Iterator[Tuple[str, Any]]:
        """Yields episodes parsed from npy files in each sequence folder."""
        seq_dirs = sorted(glob.glob(os.path.join(split_path, "seq_*")))
        # T = 100  # number of steps stored per .npy file

        for seq_path in seq_dirs:
            seq_id = os.path.basename(seq_path)
            episode = []

            # Step 1: Determine file indices to iterate over, using cam_0 as reference
            ref_files = sorted(glob.glob(os.path.join(seq_path, "cam_0", "color_*_*.npy")))
            file_tuples = [
                (int(os.path.basename(f).split("_")[1]), int(os.path.basename(f).split("_")[2].split(".")[0]))
                for f in ref_files
            ]  # (start_idx, file_id) pairs

            for start_idx, file_id in file_tuples:
                t_index = str(start_idx)

                # Step 2: Load batch arrays for each modality (loaded once per 100 steps)
                imgs = {
                    cam_id: np.load(os.path.join(seq_path, f"cam_{cam_id}", f"color_{t_index}_{file_id}.npy"))
                    for cam_id in range(4)
                }
                depths = {
                    cam_id: np.load(os.path.join(seq_path, f"cam_{cam_id}", f"depth_{t_index}_{file_id}.npy"))
                    for cam_id in range(4)
                }
                bubble_force = {
                    b_id: np.load(os.path.join(seq_path, f"bubble_{b_id}", f"force_{t_index}_{file_id}.npy"))
                    for b_id in [1, 2]
                }
                bubble_depth = {
                    b_id: np.load(os.path.join(seq_path, f"bubble_{b_id}", f"depth_{t_index}_{file_id}.npy"))
                    for b_id in [1, 2]
                }
                bubble_flow = {
                    b_id: np.load(os.path.join(seq_path, f"bubble_{b_id}", f"raw_flow_{t_index}_{file_id}.npy"))
                    for b_id in [1, 2]
                }
                ee_states = np.load(os.path.join(seq_path, "ee_states", f"ee_states_{t_index}_{file_id}.npy"))
                joint_states = np.load(os.path.join(seq_path, "joint_states", f"joint_states_{t_index}_{file_id}.npy"))
                
                T = imgs[0].shape[0]
                
                # Ensure all modalities have the same time dimension
                assert all(imgs[cid].shape[0] == T for cid in range(4)), "Image time dimension mismatch"
                assert all(depths[cid].shape[0] == T for cid in range(4)), "Depth time dimension mismatch"
                assert all(bubble_force[b_id].shape[0] == T for b_id in [1, 2]), "Bubble force time dimension mismatch"
                assert all(bubble_depth[b_id].shape[0] == T for b_id in [1, 2]), "Bubble depth time dimension mismatch" 

                # Step 3: Iterate over all time steps within this npy file
                for t in range(T):
                    import pdb; pdb.set_trace()  # Debugging breakpoint
                    obs = {
                        'images': {f'cam_{cid}': imgs[cid][t].astype(np.uint8) for cid in range(4)},
                        'depths': {f'cam_{cid}': depths[cid][t].astype(np.float32) for cid in range(4)},
                        'ee_pose': ee_states[t].astype(np.float32),
                        'joint_positions': joint_states[t].astype(np.float32),
                    }

                    for b_id in [1, 2]:
                        obs[f'bubble_{b_id}_force'] = bubble_force[b_id][t].astype(np.float32)
                        obs[f'bubble_{b_id}_depth'] = bubble_depth[b_id][t].astype(np.float32)
                        obs[f'bubble_{b_id}_flow'] = bubble_flow[b_id][t].reshape((FLOW_H, FLOW_W)).astype(np.float32)

                    step_idx = len(episode)
                    is_last_step = (start_idx, file_id) == file_tuples[-1] and t == T - 1
                    step = {
                        'observation': obs,
                        'action': np.zeros((JOINT_DIM,), dtype=np.float32),
                        'discount': 1.0,
                        'reward': float(is_last_step),
                        'is_first': step_idx == 0,
                        'is_last': is_last_step,
                        'is_terminal': is_last_step,
                    }
                    episode.append(step)

            # Step 4: Yield the complete episode for this sequence
            yield seq_id, {
                'steps': episode,
                'episode_metadata': {
                    'seq_id': seq_id,
                    'path': seq_path,
                }
            }
            