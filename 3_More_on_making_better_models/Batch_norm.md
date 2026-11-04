Input: a mini-batch of activations x₁, x₂, …, xₘ at some layer

Training
Compute the batch mean: μ = (1/m) Σ xᵢ
Compute the batch variance: σ² = (1/m) Σ (xᵢ − μ)²
Normalize each activation: x̂ᵢ = (xᵢ − μ) / √(σ² + ε)
ε is a tiny constant that prevents division by zero.
Scale and shift: yᵢ = γ · x̂ᵢ + β
γ and β are learnable parameters, trained by gradient descent.
They let the network undo the normalization if that turns out to be optimal.
Update running statistics: running = momentum · running + (1 − momentum) · batch (Do this for both the mean and the variance.)


Inference
Skip the batch computation and use the stored running mean and variance in step 3.
Output becomes deterministic and independent of other samples in the batch.


Why it works:

The original paper argued it reduces internal covariate shift, meaning each layer sees a more stable input distribution as earlier layers update.
Modern analysis suggests the real benefit is a smoother loss landscape.
Practical effects:
Allows larger learning rates
    Faster convergence
    Less sensitivity to weight initialization
    Mild regularization from the noise in batch statistics


Limitation:

Performance depends on batch size, so small batches give noisy statistics.
For NLP and sequence models, layer normalization is usually preferred. It computes statistics across the features of a single sample rather than across the batch.