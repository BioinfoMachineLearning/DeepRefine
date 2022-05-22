import logging
import os
import pickle
import shutil
import textwrap
from itertools import combinations
from pathlib import Path
from typing import Dict, Any

import atom3.database as db
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from biopandas.pdb import PandasPdb
from dgl.data import DGLDataset
from parallel import submit_jobs
from torch.utils.data import DataLoader

from project.modules.deeprefine_lit_modules import LitPSR
from project.modules.segnn.o3_transform import O3Transform
from project.utils.deeprefine_utils import collect_args, process_args, construct_pl_logger, construct_strategy, \
    process_pdb_into_graph, pyg_psr_collate, dgl_psr_collate, make_pdb_from_coords, get_interfacing_atom_indices
from project.utils.deeprefine_utils import tidy_up_and_update_pdb_file
from project.utils.segnn.utils import convert_dgl_graph_to_pyg_graph


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepRefine (https://github.com/BioinfoMachineLearning/DeepRefine):
# -------------------------------------------------------------------------------------------------------------------------------------
class InputDataset(DGLDataset):
    r"""A temporary Dataset for processing and presenting an input protein as a single Python dictionary.

    Parameters
    ----------
    input_dataset_dir: str
        The directory in which the input data will be processed. Default: 'datasets/Input'.
    atom_selection_type: str
        Which type(s) of atoms to select for graph construction and usage. Default: `all_atom`.
    knn: int
        How many nearest neighbors to which to connect a given node. Default: 20.
    idt: float
        Distance threshold under which to consider a pair of inter-chain atoms as interfacing. Default: 8.0.
    graph_return_format: str
        Which graph format to return for each input example. Default: `dgl`.
    lmax_attr: int
        Max degree of geometric attribute embedding within the SEGNN model. Default: 3.
    force_reload: bool
        Whether to reload the dataset. Default: False.
    verbose: bool
        Whether to print out progress information. Default: False.

    Notes
    -----
    The input protein will be preprocessed into local storage first.

    Examples
    --------
    >>> # Get dataset
    >>> input_data = InputDataset()
    >>>
    >>> len(input_data)
    1
    """

    def __init__(self,
                 input_dataset_dir=os.path.join('datasets', 'Input'),
                 output_dataset_dir=os.path.join('datasets', 'Output'),
                 atom_selection_type='all_atom',
                 knn=20,
                 idt=8.0,
                 graph_return_format='dgl',
                 num_workers=1,
                 lmax_attr=3,
                 force_reload=False,
                 verbose=False):
        self.input_dataset_dir = input_dataset_dir
        self.output_dataset_dir = output_dataset_dir
        self.atom_selection_type = atom_selection_type
        self.knn = knn
        self.idt = idt
        self.graph_return_format = graph_return_format
        self.num_workers = num_workers

        self.o3_transform = O3Transform(lmax_attr)
        self.ca_only = atom_selection_type == 'ca_atom'

        self.prot_dicts = {}

        self.input_filepaths = []
        for item in os.listdir(self.input_dataset_dir):
            item_path = os.path.join(self.input_dataset_dir, item)
            if os.path.isfile(item_path) and '_plddt' not in item and '.dill' not in item:
                # Ensure the input filenames contain the .pdb file extension
                if not os.path.splitext(item_path)[-1] == '.pdb':
                    new_item_path = item_path + '.pdb'
                    shutil.move(item_path, new_item_path)
                    item_path = new_item_path

                self.input_filepaths.append(item_path)
        self.input_filepaths.sort()

        super(InputDataset, self).__init__(name='InputDataset',
                                           raw_dir=input_dataset_dir,
                                           force_reload=force_reload,
                                           verbose=verbose)
        logging.info(f"Loading proteins for prediction, input directory: {self.input_dataset_dir}")

    def download(self):
        pass

    def process(self):
        """Process each protein into a prediction-ready dictionary using multiple CPU workers."""
        inputs = []
        for i, fp in enumerate(self.input_filepaths):
            # Only process inputs that have not already been saved to storage as a graph
            output_fp = os.path.join(self.output_dataset_dir, os.path.splitext(os.path.split(fp)[-1])[0] + '.dill')

            if not os.path.exists(output_fp):
                inputs.append((fp, i, True))
            else:
                # Determine whether to process new graph or overwrite existing one
                with open(output_fp, 'rb') as f:
                    prot_dict = pickle.load(f)

                dihedral_angles = prot_dict['graph'].ndata['dihedral_angles']
                dihedral_angles_given = torch.any(dihedral_angles).sum() > 0
                case_1 = dihedral_angles_given and self.atom_selection_type == 'all_atom'
                case_2 = not dihedral_angles_given and self.atom_selection_type == 'ca_atom'
                cached_graph_in_different_atom_format = case_1 or case_2

                if cached_graph_in_different_atom_format:
                    inputs.append((fp, i, True))
                else:
                    inputs.append((fp, i, False))
        submit_jobs(self.process_filepath_into_prot_dict, inputs, self.num_workers)

    def has_cache(self):
        pass

    def prepare_protein_dict_for_forward(self, protein_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Update a given protein dictionary in preparation for a forward pass through a network."""
        # Combine all distinct node features together into a single node feature tensor
        protein_dict['graph'].ndata['f'] = torch.cat((
            protein_dict['graph'].ndata['atom_type'],
            protein_dict['graph'].ndata['surf_prox']
        ), dim=1)

        if self.ca_only:
            # Add dihedral angles for Ca-atom graphs
            protein_dict['graph'].ndata['f'] = torch.cat((
                protein_dict['graph'].ndata['f'],
                protein_dict['graph'].ndata['dihedral_angles']
            ), dim=1)

        # Combine all distinct edge features into a single edge feature tensor
        protein_dict['graph'].edata['f'] = torch.cat((
            protein_dict['graph'].edata['pos_enc'],
            protein_dict['graph'].edata['in_same_chain'],
            protein_dict['graph'].edata['rel_geom_feats']
        ), dim=1)

        if not self.ca_only:
            # Add bond types for all-atom graphs
            protein_dict['graph'].edata['f'] = torch.cat((
                protein_dict['graph'].edata['f'],
                protein_dict['graph'].edata['bond_type']
            ), dim=1)

        # Return the requested protein dictionary after updates
        return protein_dict

    def process_filepath_into_prot_dict(self, input_filepath: str, input_idx: int, process: bool = True):
        """Process a protein into a prediction-ready dictionary."""
        # Establish the new dictionary's filename and path
        output_filepath = os.path.join(
            self.output_dataset_dir, os.path.splitext(os.path.split(input_filepath)[-1])[0] + '.dill'
        )

        # Save the constructed graph to storage as a preprocessing step
        if process:
            # Tidy-up the input PDB
            tidy_up_and_update_pdb_file(input_filepath)

            # Process the unprocessed protein
            print(f'Processing PDB {input_filepath} into DGLGraph')
            graph = process_pdb_into_graph(input_filepath,
                                           self.atom_selection_type,
                                           self.knn,
                                           self.idt)

            # Assemble all valid combinations of chains for scoring chain-pair body intersection losses
            unique_chain_ids = np.unique(graph.ndata['chain_id'])
            chain_combinations = torch.tensor(list(combinations(unique_chain_ids, r=2))).float()

            # Organize the input graph and its associated metadata as a dictionary
            prot_dict = {
                'protein': db.get_pdb_name(input_filepath),
                'graph': graph,
                'chain_combinations': chain_combinations,
                'filepath': input_filepath
            }

            # Define graph metadata for the entire forward-pass pipeline
            prot_dict = self.prepare_protein_dict_for_forward(prot_dict)

            # Ascertain number of input node and edge features
            self.number_of_node_features = prot_dict['graph'].ndata['f'].shape[1]
            self.number_of_edge_features = prot_dict['graph'].edata['f'].shape[1]

            # Save the new dictionary to storage
            with open(output_filepath, 'wb') as f:
                pickle.dump(prot_dict, f)

        # Cache the input protein for later retrieval
        self.prot_dicts[str(input_idx)] = output_filepath

    def __getitem__(self, idx: int):
        """Return requested protein to DataLoader."""
        filepath = self.prot_dicts[str(idx)]

        # Load in protein dictionary
        with open(filepath, 'rb') as f:
            prot_dict = pickle.load(f)

        # Convert between graph formats, as requested
        if self.graph_return_format == 'pyg':
            prot_dict['graph'] = convert_dgl_graph_to_pyg_graph(prot_dict['graph'], self.ca_only, labels=False)
            prot_dict['graph'] = self.o3_transform(prot_dict['graph'])

        return prot_dict

    def __len__(self) -> int:
        """Number of proteins in the dataset."""
        return len(self.input_filepaths)

    @property
    def num_node_features(self) -> int:
        """Retrieve the number of node feature values after encoding them."""
        return 31 if self.ca_only else 39

    @property
    def num_edge_features(self) -> int:
        """Retrieve the number of edge feature values after encoding them."""
        return 14 if self.ca_only else 15

    @property
    def raw_path(self) -> str:
        """Directory in which to locate raw proteins."""
        return self.raw_dir


def main(args):
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
            performing_quality_estimation = True
            if performing_quality_estimation:
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
    # Arguments
    # -----------
    # Collect all arguments
    parser = collect_args()

    # Let the model add what it wants
    parser = LitPSR.add_model_specific_args(parser)

    # Add argument(s) required for model inference
    parser.add_argument('--perform_pos_refinement', action='store_true', dest='perform_pos_refinement')

    # Parse all known and unknown arguments after adding those that are model specific
    args, unparsed_argv = parser.parse_known_args()

    # Build the strategy for Lightning to use
    find_unused = args.qa_loss_weight == 0.0  # When not using QA loss, enable garbage collection of unused parameters
    strategy = construct_strategy(args.device_strategy, find_unused)  # Construct training strategy for Lightning to use

    # Set Lightning-specific parameter values before constructing Trainer instance
    args.max_epochs = args.num_epochs
    args.profiler = args.profiler_method
    args.accelerator = args.device_type
    args.strategy = strategy
    args.devices = 1  # Enforce inferences to take place on a single GPU
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

    # Begin execution of model training with given args
    main(args)
