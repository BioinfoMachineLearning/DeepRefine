# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from AlphaFold (https://github.com/deepmind/alphafold):
# -------------------------------------------------------------------------------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2022 University of Missouri-Columbia Bioinformatics & Machine Learning (BML) Lab.

"""Docker launch script for DeepRefine docker image."""

import os
import signal
from pathlib import Path
from typing import Tuple

import docker
from absl import app
from absl import flags
from absl import logging
from docker import types

flags.DEFINE_bool('use_gpu', True, 'Enable NVIDIA runtime to run with GPUs.')
flags.DEFINE_string('gpu_devices', 'all', 'Comma separated list of devices to pass to NVIDIA_VISIBLE_DEVICES.')
flags.DEFINE_string('docker_image_name', 'deeprefine', 'Name of DeepRefine docker image.')

flags.DEFINE_integer('num_gpus', 1, 'How many GPUs to use to make a prediction (--num_gpus 0 means use CPU instead)')
flags.DEFINE_integer('num_workers', 1, 'Number of CPU threads for loading data')
flags.DEFINE_string('input_dataset_dir', None, 'Directory in which to expect input PDB target dirs. to be stored.')
flags.DEFINE_string('output_dir', None, 'Directory in which to store generated outputs and predictions for inputs.')
flags.DEFINE_string('ckpt_dir', None, 'Directory from which to load checkpoints.')
flags.DEFINE_string('ckpt_name', None, 'Name of trained model checkpoint to use.')
flags.DEFINE_string('atom_selection_type', None, 'Type(s) of atoms to use in graphs.')
flags.DEFINE_integer('seed', None, 'Seed for NumPy and PyTorch.')
flags.DEFINE_string('nn_type', None, 'Type of neural network to use for forward propagation.')
flags.DEFINE_string('graph_return_format', None, 'Which graph format to return.')
flags.DEFINE_bool('perform_pos_refinement', False, 'Whether to refine node positions during inference.')

FLAGS = flags.FLAGS

_ROOT_MOUNT_DIRECTORY = '/mnt/'


def _create_mount(mount_name: str, path: str, execute=True,
                  type='bind', read_only=True, same_level=False) -> Tuple[types.Mount, str]:
    path = os.path.abspath(path)
    source_path = path if same_level else os.path.dirname(path)
    target_path = os.path.join(_ROOT_MOUNT_DIRECTORY, mount_name)
    logging.info('Mounting %s -> %s', source_path, target_path)
    mount = types.Mount(target_path, source_path, type=type, read_only=read_only) if execute else None
    return mount, os.path.join(target_path, os.path.basename(path))


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    #### USER CONFIGURATION ####

    # Number of GPUs to use for predictions (num_gpus=0 means use CPU instead)
    num_gpus = FLAGS.num_gpus

    # Path to directory for storing input proteins.
    input_dataset_dir = Path(FLAGS.input_dataset_dir).absolute().as_posix()

    # Path to directory for storing generated outputs and predictions.
    output_dir = Path(FLAGS.output_dir).absolute().as_posix()

    # Path to directory containing trained models (i.e., PyTorch LightningModule checkpoints).
    ckpt_dir = Path(FLAGS.ckpt_dir).absolute().as_posix()

    # Name of trained model to use for predictions (i.e., PyTorch LightningModule checkpoint).
    ckpt_name = FLAGS.ckpt_name

    # Type(s) of atoms to use in graphs.
    atom_selection_type = FLAGS.atom_selection_type

    # Seed for NumPy and PyTorch.
    seed = FLAGS.seed

    # Type of neural network to use for forward propagation.
    nn_type = FLAGS.nn_type

    # Which graph format to return.
    graph_return_format = FLAGS.graph_return_format

    # Whether to refine node positions during inference.
    perform_pos_refinement = FLAGS.perform_pos_refinement

    #### END OF USER CONFIGURATION ####

    mounts = []
    command_args = []

    # Set number of GPUs to use for predictions
    command_args.append(f'--num_gpus={num_gpus}')

    # Mount directory for storing input proteins
    mount, target_path = _create_mount('Input', input_dataset_dir, same_level=True, read_only=False)
    mounts.append(mount)
    command_args.append(f'--input_dataset_dir={os.path.dirname(target_path)}')

    # Mount directory for storing outputs
    mount, target_path = _create_mount('Output', output_dir, same_level=True, read_only=False)
    mounts.append(mount)
    command_args.append(f'--output_dir={os.path.dirname(target_path)}')

    # Mount directory for storing requested checkpoint to be used for prediction
    mount, target_path = _create_mount('checkpoints', ckpt_dir)
    mounts.append(mount)
    command_args.append(f'--ckpt_dir={target_path}')
    command_args.append(f'--ckpt_name={ckpt_name}')

    # Set type of atoms to use during prediction
    command_args.append(f'--atom_selection_type={atom_selection_type}')

    # Set random seed to use during prediction
    command_args.append(f'--seed={seed}')

    # Set type of neural network to use for forward propagation
    command_args.append(f'--nn_type={nn_type}')

    # Set type of graph representation to use during prediction
    command_args.append(f'--graph_return_format={graph_return_format}')

    # Set position refinement mode to use during prediction
    command_args.append(f'--perform_pos_refinement={perform_pos_refinement}')

    client = docker.from_env()
    container = client.containers.run(
        image=FLAGS.docker_image_name,
        command=command_args,
        runtime='nvidia' if FLAGS.use_gpu else None,
        remove=True,
        detach=True,
        mounts=mounts,
        environment={
            'NVIDIA_VISIBLE_DEVICES': FLAGS.gpu_devices,
        })

    # Add signal handler to ensure CTRL+C also stops the running container.
    signal.signal(signal.SIGINT,
                  lambda unused_sig, unused_frame: container.kill())

    for line in container.logs(stream=True):
        logging.info(line.strip().decode('utf-8'))


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'input_dataset_dir',
        'output_dir',
        'ckpt_dir',
        'ckpt_name',
        'atom_selection_type',
        'seed',
        'nn_type',
        'graph_return_format'
    ])
    app.run(main)
