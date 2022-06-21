# MLMI-2022
Repository to hold the codebase for the MLMI 2022 Workshop submission titled "Neural State-Space Modeling with Latent Causal-Effect Disentanglement"

## Data generation
We provide the scripts that generate the native and intervention dynamics sets. Follow the README under <code>Data Generation/</code> for additional information.

## Saved models
Each model shown has a pre-trained checkpoint available to reload, saved each under <code>checkpoints</code> in their individually named folders in <code>experiments</code>. To load in a trained model, set the argparse argument <code>--ckpt</code> to '1'.

## Training
Each model is a separate Pytorch-Lightning training class that handles dataloading and training, as well as checkpoint and visualizing training samples over epochs. To run a model, simply run the file of the model desired (e.g. <code> python3 ode_vae</code>. Version denotes which dataset to use, which is either <code>Normal</code> or <code>Pacing</code>.
