# Data Modules

Data modules are specific to the framework Lightning AI. To learn more about
them check the [docs](https://lightning.ai/docs/pytorch/stable/data/datamodule.html).

# `MaskKeypointDataset` class

This class is responsible for masking and feeding the information needed during
the feasibility study and pretraining stage. The only difference between the 
`PretrainMaskKeypointDataset` and the `MaskKeypointDataset` is that the former
handles both hands.

# Future lines of work

Currently, the `PretrainMaskKeypointDataset` masks both hands independently 
until the number of required masked frames is met. It could be changed so 
masking either one contributes towards reaching the number of masked frames
requirement.