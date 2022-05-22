import logging
import os
import shutil
import textwrap
from pathlib import Path

import atom3.database as db
import pandas as pd
import pytorch_lightning as pl
import torch
from absl import flags, app
from biopandas.pdb import PandasPdb
from torch.utils.data import DataLoader

from project.lit_model_predict import InputDataset
from project.modules.deeprefine_lit_modules import LitPSR
from project.utils.deeprefine_utils import collect_args, process_args, construct_pl_logger, construct_strategy, \
    pyg_psr_collate, dgl_psr_collate, make_pdb_from_coords, get_interfacing_atom_indices

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
flags.DEFINE_bool('perform_pos_refinement', None, 'Whether to refine node positions during inference.')

FLAGS = flags.FLAGS


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepRefine (https://github.com/BioinfoMachineLearning/DeepRefine):
# -------------------------------------------------------------------------------------------------------------------------------------
def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # -----------
    # ArgParse
    # -----------
    # Collect all arguments
    parser = collect_args()

    # Let the model add what it wants
    parser = LitPSR.add_model_specific_args(parser)

    # Parse all known and unknown arguments after adding those that are model specific
    args, unparsed_argv = parser.parse_known_args()

    # Use Abseil's latest Docker arguments
    args.num_devices = FLAGS.num_gpus
    args.num_workers = FLAGS.num_workers
    args.input_dataset_dir = FLAGS.input_dataset_dir
    args.output_dir = FLAGS.output_dir
    args.ckpt_dir = FLAGS.ckpt_dir
    args.ckpt_name = FLAGS.ckpt_name
    args.atom_selection_type = FLAGS.atom_selection_type
    args.seed = FLAGS.seed
    args.nn_type = FLAGS.nn_type
    args.graph_return_format = FLAGS.graph_return_format
    args.perform_pos_refinement = FLAGS.perform_pos_refinement in [True, 'True', 'true']

    # Build the strategy for Lightning to use
    find_unused = args.qa_loss_weight == 0.0  # When not using QA loss, enable garbage collection of unused parameters
    strategy = construct_strategy(args.device_strategy,
                                  find_unused)  # Construct training strategy for Lightning to use

    # Set Lightning-specific parameter values before constructing Trainer instance
    args.max_epochs = args.num_epochs
    args.profiler = args.profiler_method
    args.accelerator = args.device_type
    args.strategy = strategy
    args.devices = args.num_devices  # Enforce inferences to take place on a single GPU
    args.num_nodes = 1  # Enforce inferences to take place on a single node
    args.precision = args.gpu_precision
    args.accumulate_grad_batches = args.accum_grad_batches
    args.gradient_clip_val = args.grad_clip_val
    args.gradient_clip_algo = args.grad_clip_algo if args.grad_clip_val > 0 else None
    args.check_val_every_n_epoch = args.check_val_every_n_train_epochs
    args.deterministic = False  # If available for local hardware, make LightningModule's training deterministic
    args.detect_anomaly = False  # Inspect for anomalies during training (e.g., NaNs or infs found during forward())

    # Finalize all arguments as necessary
    args = process_args(args)

    # -----------
    # Test Args
    # -----------
    predict_batch_size = 1  # Enforce batch_size=1 to account for large proteins in the input dataset
    ca_only = args.atom_selection_type == 'ca_atom'

    # ------------
    # Model
    # ------------
    # Assemble a dictionary of model arguments
    dict_args = vars(args)
    use_wandb_logger = args.logger_name.lower() == 'wandb'  # Determine whether the user requested to use WandB

    # Craft an experiment name for the current inference run
    args.experiment_name = f'LitPSR' \
                           f'-m{args.nn_type.strip().upper()}' \
                           f'-b{args.batch_size}' \
                           f'-l{args.num_layers}' \
                           f'-n{args.num_hidden_channels}' \
                           f'-e{args.num_hidden_channels}' \
        if not args.experiment_name \
        else args.experiment_name

    # ------------
    # Checkpoint
    # ------------
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    ckpt_provided = args.ckpt_name != ''
    assert ckpt_provided, 'A checkpoint filename must be provided'

    # ------------
    # Trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # ------------
    # Logger
    # ------------
    pl_logger = construct_pl_logger(args)  # Log everything to an external logger
    trainer.logger = pl_logger  # Assign specified logger (e.g., TensorBoardLogger) to Trainer instance

    # ------------
    # Restore
    # ------------
    if use_wandb_logger and args.ckpt_name != '' and not os.path.exists(ckpt_path):
        # If using WandB, download checkpoint artifact from their servers if the checkpoint is not already local
        checkpoint_reference = f'{args.entity}/{args.project_name}/model-{args.run_id}:best'
        artifact = trainer.logger.experiment.use_artifact(checkpoint_reference, type='model')
        artifact_dir = artifact.download()
        ckpt_filepath = Path(artifact_dir) / 'model.ckpt'
    else:
        # Otherwise, use the user's provided checkpoint filepath to load the checkpoint
        assert ckpt_provided and os.path.exists(ckpt_path), 'A valid checkpoint filepath must be provided'
        ckpt_filepath = ckpt_path

    # Load provided checkpoint
    model = LitPSR.load_from_checkpoint(ckpt_filepath,
                                        use_wandb_logger=use_wandb_logger,
                                        nn_type=args.nn_type,
                                        tmscore_exec_path=args.tmscore_exec_path,
                                        dockq_exec_path=args.dockq_exec_path,
                                        galaxy_exec_path=args.galaxy_exec_path,
                                        galaxy_home_path=args.galaxy_home_path,
                                        use_ext_tool_only=args.use_ext_tool_only,
                                        experiment_name=dict_args['experiment_name'],
                                        strict=False)

    # -----------
    # Target Loop
    # -----------
    for item in os.listdir(args.input_dataset_dir):
        in_item_path = os.path.join(args.input_dataset_dir, item)
        out_item_path = os.path.join(args.output_dir, item)
        os.makedirs(out_item_path, exist_ok=True)
        if os.path.isdir(in_item_path):
            # -----------
            # Input
            # -----------
            collate_fn = pyg_psr_collate if args.graph_return_format == 'pyg' else dgl_psr_collate
            input_dataset = InputDataset(input_dataset_dir=in_item_path,
                                         output_dataset_dir=out_item_path,
                                         atom_selection_type=args.atom_selection_type,
                                         knn=20,
                                         idt=args.idt,
                                         graph_return_format=args.graph_return_format,
                                         num_workers=args.num_workers,
                                         lmax_attr=args.lmax_attr)
            input_dataloader = DataLoader(input_dataset,
                                          batch_size=predict_batch_size,
                                          shuffle=False,
                                          num_workers=0,
                                          collate_fn=collate_fn)

            # -----------
            # Prediction
            # -----------
            # Predict with a trained model using the provided input dataloader
            perform_quality_estimation = True
            if perform_quality_estimation:
                casp_preds_col = []
                prediction_results = trainer.predict(model=model, dataloaders=input_dataloader)
                for updated_graph, filepaths in prediction_results:
                    filepath = filepaths[0]
                    if args.perform_pos_refinement:
                        # Build all-atom PDB file(s) from Ca atom trace PDB, and save results to storage
                        output_pdb_filepath = os.path.join(out_item_path, os.path.split(filepath)[-1])
                        if not os.path.exists(output_pdb_filepath):
                            # Ensure the input PDB file is present in the output directory to output new PDB coordinates
                            input_pdb_filepath = os.path.join(in_item_path, os.path.split(filepath)[-1])
                            shutil.copy(input_pdb_filepath, output_pdb_filepath)
                        make_pdb_from_coords(
                            updated_graph.pos if args.graph_return_format == 'pyg' else updated_graph.ndata['x_pred'],
                            output_pdb_filepath,
                            'mm',
                            args.atom_selection_type == 'ca_atom'
                        )

                    # Derive data from the input PDB file
                    pdb = PandasPdb().read_pdb(filepath)
                    atoms_df = pdb.df['ATOM']
                    atoms_df.reset_index(drop=True, inplace=True)
                    ca_atoms_df = atoms_df[atoms_df['atom_name'] == 'CA']
                    num_res = len(ca_atoms_df)
                    ca_atom_idx = torch.LongTensor(ca_atoms_df.index)

                    # Assemble global structure quality scores predicted by the model
                    qa_pred = updated_graph.ndata if args.graph_return_format == 'pyg' else updated_graph.gdata['q']
                    g_pred_node_feats = qa_pred.reshape(-1, 1)
                    if not ca_only:
                        g_pred_node_feats = g_pred_node_feats[ca_atom_idx, :]
                    global_quality_pred = torch.clamp(
                        g_pred_node_feats,
                        min=0.0,
                        max=1.0
                    )  # Ensure pLDDT is in [0, 1]
                    g_quality_pred_df = pd.DataFrame(global_quality_pred.cpu().numpy(), columns=['Predicted LDDT-Ca'])
                    g_quality_pred_mean_val = g_quality_pred_df.sum(axis=0).values.squeeze() / num_res
                    g_quality_pred_df.loc[len(g_quality_pred_df.index)] = [g_quality_pred_mean_val]

                    # Identify interfacing residues, and represent them with their Ca atoms
                    interface_distance_threshold = 8.0  # Note: This quantity is measured in Angstrom
                    _, interfacing_atom_mapping = get_interfacing_atom_indices(
                        atoms_df, atoms_df.index, interface_distance_threshold, return_partners=True
                    )
                    interfacing_ca_atom_indices = set()
                    for key in interfacing_atom_mapping.keys():
                        first_atom = atoms_df.loc[key, :]
                        # Follow CAPRI's definition of interface by looking for inter-chain res. w/ CB-CB distance < 8.0
                        is_cb_first = first_atom['atom_name'] == 'CB'
                        is_gly_ca_first = (first_atom['residue_name'] == 'GLY' and first_atom['atom_name'] == 'CA')
                        if is_cb_first or is_gly_ca_first:
                            # Account for Glycine not having carbon-beta (CB) atom by substituting it with Ca atom
                            interfacing_atom_ids = interfacing_atom_mapping[key]
                            for i in interfacing_atom_ids:
                                second_atom = atoms_df.loc[i, :]
                                is_cb_second = second_atom['atom_name'] == 'CB'
                                is_gly_ca_second = (
                                        second_atom['residue_name'] == 'GLY' and second_atom['atom_name'] == 'CA')
                                if is_cb_second or is_gly_ca_second:
                                    first_res_ca_atom = atoms_df[
                                        (atoms_df['chain_id'] == first_atom['chain_id']) &
                                        (atoms_df['residue_number'] == first_atom['residue_number'].item()) &
                                        (atoms_df['atom_name'] == 'CA')
                                        ]
                                    second_res_ca_atom = atoms_df[
                                        (atoms_df['chain_id'] == second_atom['chain_id']) &
                                        (atoms_df['residue_number'] == second_atom['residue_number'].item()) &
                                        (atoms_df['atom_name'] == 'CA')
                                        ]
                                    interfacing_ca_atom_indices.add(first_res_ca_atom.index.item())
                                    interfacing_ca_atom_indices.add(second_res_ca_atom.index.item())

                    interfacing_ca_atom_indices = list(interfacing_ca_atom_indices)
                    interfacing_ca_atom_indices.sort()
                    interfacing_ca_atom_indices = torch.LongTensor(interfacing_ca_atom_indices)
                    interfacing_ca_atoms_df = atoms_df.loc[pd.Index(interfacing_ca_atom_indices.cpu().numpy()), :]

                    if ca_only:
                        i_ca_atoms_df = pd.merge(ca_atoms_df, interfacing_ca_atoms_df)
                        interfacing_ca_atom_indices = torch.LongTensor(ca_atoms_df.reset_index(drop=True)[
                                                                           ca_atoms_df.reset_index(drop=True)[
                                                                               'atom_number'].isin(
                                                                               i_ca_atoms_df.reset_index(drop=True)[
                                                                                   'atom_number']
                                                                           )
                                                                       ].index)

                    # Assemble interface structure quality scores predicted by the model
                    i_pred_node_feats = qa_pred.clone().reshape(-1, 1)
                    if ca_only:
                        i_pred_node_feats = i_pred_node_feats[interfacing_ca_atom_indices, :]
                    else:
                        i_pred_node_feats = i_pred_node_feats[interfacing_ca_atom_indices, :]
                    i_quality_pred = torch.clamp(
                        i_pred_node_feats,
                        min=0.0,
                        max=1.0
                    )  # Ensure pLDDT is in [0, 1]
                    i_quality_pred_df = pd.DataFrame(i_quality_pred.cpu().numpy(), columns=['Predicted LDDT-Ca'])
                    i_quality_pred_mean_val = i_quality_pred_df.sum(axis=0).values.squeeze() / len(i_quality_pred_df)

                    # Aggregate CASP predictions
                    global_mqs = 1 - g_quality_pred_mean_val
                    interface_mqs = 1 - i_quality_pred_mean_val
                    i_res_qs = [
                        f'{tpl.chain_id}{tpl.residue_number}:{i_quality_pred[i].item()}'
                        for i, tpl in enumerate(interfacing_ca_atoms_df.itertuples(index=False))
                    ]

                    casp_preds_col.append([
                        os.path.split(filepath)[-1], global_mqs.item(), interface_mqs.item(), *i_res_qs
                    ])

                    # Record structure quality scores predicted by the model
                    quality_pred_filename_postfix = '_refined' if args.perform_pos_refinement else ''
                    quality_pred_output_filepath = os.path.join(
                        out_item_path,
                        f'{os.path.splitext(os.path.basename(filepath))[0]}{quality_pred_filename_postfix}_plddt.csv'
                    )
                    g_quality_pred_df.to_csv(quality_pred_output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    # -----------
    # Abseil
    # -----------
    flags.mark_flags_as_required([
        'input_dataset_dir',
        'output_dir',
        'ckpt_dir',
        'ckpt_name',
        'atom_selection_type',
        'seed',
        'nn_type',
        'graph_return_format',
        'perform_pos_refinement'
    ])
    app.run(main)
