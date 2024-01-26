In the mnist_model_bias_eraser, there are files(model, code, loader) adapting the dependent learning technique to another version of evaluator.

Even though this evaluator takes two inputs(x1 for original and x2 for reconstructed),
the presented design intentionally makes it difficult to distinguish
between the original(x1) and the reconstructed(x2), as it is the case in the double slit experiment.

I hope this design solves a possible concern that the original evaluator would have been exhibiting biases toward the reconstructed data.


The advantage: eliminating bias

The disadvantage: lost unsupervisedness(as it become self-supervised)


The file Evaluators.py contains types of evaluator that can be used at hands.

Here is ChatGPT(GPT-4)'s opinion on the list of evaluators.

Summary

Complexity: Increases from Evaluator 1 to Evaluator 4, with more sophisticated processing of dual inputs in the latter evaluators.
Bias Reduction: Evaluators 3 and 4 are specifically designed with bias reduction in mind, using techniques to intertwine features from dual inputs.

Feature Preservation: Evaluator 4 seems to strike the best balance between preserving individual input features and reducing bias, due to its more layered and nuanced architecture.

Use Cases: Evaluator 1 is best for simple tasks, Evaluator 2 for tasks requiring feature blending, Evaluator 3 for focused bias reduction, and Evaluator 4 for complex tasks needing both feature preservation and bias reduction.
