In the mnist_model_bias_eraser, there are files(model, code, loader) adapting the dependent learning technique to another version of evaluator.

Even though this evaluator takes two inputs(x1 for original and x2 for reconstructed),
the presented design intentionally makes it difficult to distinguish
between the original(x1) and the reconstructed(x2), as it is the case in the double slit experiment.

I hope this design solves a possible concern that the original evaluator would have been exhibiting biases toward the reconstructed data.


The advantage: eliminating bias

The disadvantage: lost unsupervisedness(as it become self-supervised)
