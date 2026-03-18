# Model evaluation

Random note that I don't know where else to put (06/02/2026):
With the latest iteration of the model, all output score were NaN,
despite the fact that nothing changed in the model architecture or input variable types.
The root cause is yet unclear, but it's probably an ONNX internal mismatch between HPG where it was trained/converted
and lxplus where it is evaluated.
The evaluation works just fine on HPG, and turns out to also work here on lxplus
in the "weaver" conda environment, rather than the key4hep stack...
So for now can just temporarily switch environments to run the inference.
