# How the lab works and how to build on it

## Learning approach

The motivating question was: **why can a residual network remain trainable when a deep plain network loses useful gradients?** A static final prediction does not show how this happens. The lab connects a selected input to its hidden features, output, loss, parameter derivatives, and a controlled intervention on a weight.

The design grew from Hongrui Liu's preference for learning by adjusting a model and watching every relevant stage. Its main questions are:

- How does a grid of inputs change after each block, before and after training?
- Why can the plain feature cloud contract while the residual cloud remains spread out?
- Where do gradients become tiny, and does that persist as training progresses?
- What correction does a residual block actually add?
- How does changing one early or late weight change one example's loss?

Treat these as hypotheses to investigate. Start with a reproducible initialization, change one condition, and compare numerical measurements alongside pictures. A visual impression is evidence about the displayed run, not proof of a universal property.

## Model and parameter counts

The input is $x=(x_1,x_2)\in[-2,2]^2$. Both models use a tanh stem $h_0=\tanh(W_0x+b_0)$, $B$ hidden blocks, and a linear scalar head $\hat y=w^Th_B+b$. Widths are $d_0,\ldots,d_B$.

For hidden block $l=1,\ldots,B$:

$$h_l^{\mathrm{plain}}=\tanh(W_lh_{l-1}+b_l),$$

$$h_l^{\mathrm{res}}=S_lh_{l-1}+\alpha\tanh(W_lh_{l-1}+b_l),\qquad \alpha=1/\sqrt B.$$

$W_l$ has shape $d_l\times d_{l-1}$ and $b_l$ has length $d_l$. For equal widths, $S_l=I$ with no parameters. For a width change, $S_l=P_l$ is a trainable, bias-free matrix initialized as a rectangular identity. The public page calls these main weights W, biases b, and shortcut weights P.

The plain parameter count is

$$3d_0+\sum_{l=1}^{B}d_l(d_{l-1}+1)+(d_B+1).$$

The residual model adds $d_ld_{l-1}$ parameters for each width-changing shortcut. At the default **36 blocks of width 10**, both models have **4,001 parameters**: stem 30, each block 110, and head 11. Here “36 blocks” excludes the stem and output head; there are 38 trainable affine layers. The 3 × 2 preset has 27 parameters per model.

Main weights use the same seeded Gaussian draws with standard deviation $\sqrt{2/(d_{in}+d_{out})}$; biases start at zero. Adding projection shortcuts consumes no extra random draws. Thus corresponding main weights match at initialization, while the forward computations differ.

## Why residual gradients can behave differently

The local hidden-state Jacobians are

$$J_l^{\mathrm{plain}}=D_lW_l,\qquad J_l^{\mathrm{res}}=S_l+\alpha D_lW_l,$$

where $D_l$ is diagonal with entries $1-\tanh^2(W_lh_{l-1}+b_l)$. Backpropagation multiplies transposes of these Jacobians. Repeated contraction or tanh saturation can strongly attenuate gradients in the plain model. An equal-width residual block retains an identity term, and scaling the branch can keep its Jacobian closer to identity.

For a scalar illustration, multiplying 36 derivatives of 0.5 gives $0.5^{36}\approx1.46\times10^{-11}$. Multiplying 36 near-identity derivatives of 0.98 gives approximately 0.48. These are illustrative local derivatives, not a claim that either network has a constant derivative or that residual gradients cannot vanish or explode. Projection bottlenecks, large learned branches, saturation, and optimizer dynamics still matter.

The default deep tanh model is intended to make gradient transport inspectable. Raw gradient norm does not equal parameter-update norm: Adam uses per-parameter first and second moments. A smoother loss history also does not establish a globally smoother loss landscape. Useful background is the original [ResNet paper](https://arxiv.org/abs/1512.03385), [identity-mapping analysis](https://arxiv.org/abs/1603.05027), and [loss-landscape visualization paper](https://arxiv.org/abs/1712.09913). This lab uses a simpler dense tanh model, without the convolutional and normalization details of those image models.

## Target, data, and optimizer

The target is a sum of two Gaussian hills, one Gaussian valley, and a saddle term:

$$f(x,y)=1.3e^{-2.3[(x+0.8)^2+(y-0.5)^2]}+0.9e^{-3[(x-0.9)^2+(y+0.6)^2]}-1.05e^{-2.8[(x-0.45)^2+(y-0.9)^2]}+0.18xy.$$

Training examples are uniformly sampled on $[-2,2]^2$ with data seed 1234. Both networks receive identical shuffled batches of 128; the shuffle seed depends on the initialization seed. The default dataset has 1,024 examples. The optimization objective is mean squared error $N^{-1}\sum_i(\hat y_i-y_i)^2$ on the current batch. Only the final scalar output is supervised.

The public UI uses Adam with learning rate 0.003, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$ and bias-corrected moments. The engine also implements plain SGD for code-level experiments, but no optimizer selector is exposed in this release. There is no gradient clipping, weight decay, dropout, batch normalization, or early stopping.

Each update records full-training-set MSE and fixed-grid MSE. The latter uses 625 points on a 25 × 25 grid. Because users repeatedly inspect it, call it an evaluation grid rather than a final untouched benchmark test set.

## Visualization choices

| View | Implementation and interpretation |
| --- | --- |
| Hidden-feature clouds | SVG; 289 fixed probe inputs. Grid edges preserve input adjacency. The points are activations, not parameters. |
| Live PCA | Center pooled plain/residual features, compute covariance, use a cyclic Jacobi symmetric eigensolver, project onto two leading axes. Sign alignment reduces flips but does not freeze the basis. |
| Residual correction | Plot the actual scaled branch activation $\alpha\tanh(Wh+b)$, with its own PCA and axes. At changing widths, subtracting unprojected input from output would be incorrect. |
| Prediction surfaces | Canvas 2D renders projected 3D wireframes, synchronized camera rotation, shared output range, actual training points, and the selected point. No WebGL dependency. |
| Heatmaps | SVG matrix cells with signed diverging colors and numeric limits. Plain/residual share bounds within parameter and gradient pairs; parameter and derivative units remain distinct. |
| Loss and gradient curves | SVG, logarithmic vertical axes. Actual zeros get an adaptive floor; tiny positive gradients are not clamped to a convenient plotting threshold. |
| Exact signals | A numeric table preserves information lost by the 2D view. |
| Parameter intervention | Conditional loss curve evaluated by temporarily changing one parameter and restoring it in a `finally` block. The optimizer is untouched. |

There are no chart-library, CDN, or package downloads at runtime. The standalone HTML contains its own code and styles. Browser events update shared state; numerical caches are keyed by model identity, step, layer, and selected example as needed. Training yields between updates so the user can pause. Architecture controls are disabled during a running update sequence; reset can stop it.

The gradient source selector affects derivative inspection, not loss-history evaluation or the training data. Gradients are evaluated at the currently displayed weights on the latest batch, full training set, or single example. Main-weight Frobenius norms omit biases; RMS divides by the square root of the number of weights. Width-changing shortcut gradients have separate markers.

The one-example training button updates every trainable parameter using the current Adam state. It increments the optimizer counter and history, without appending data or advancing the normal batch cursor. It can increase that example's loss because its Adam moments reflect earlier updates. Previewing a weight performs no such training.

## Alternative approaches and tradeoffs

- **Two features everywhere:** direct coordinates are easy to interpret, but narrow layers can constrain the task. Adjustable widths plus PCA preserve a view while exposing its information loss. Joint PCA gives a common projection, but one network's greater variance can dominate it; the retained-variance labels matter.
- **Independent PCA for each network:** can reveal internal structure better, but its axes are not directly comparable. The correction plot deliberately uses this approach because it is a different quantity, with an explicit separate-scale caption.
- **An output head after each hidden layer:** gives per-layer scalar surfaces, but auxiliary training losses change the learning problem. This lab keeps one final supervised head and labels hidden plots as features.
- **A single-weight slice versus a two-dimensional landscape:** one weight permits an exact labeled intervention with cheap evaluation. Two-weight or normalized-direction surfaces would add information, but remain conditional slices and cost more evaluations.
- **Manual JavaScript backpropagation versus PyTorch:** the small CPU model can run without setup in every visitor's browser. PyTorch is useful as an independent reference and for larger CNN experiments. A future engine replacement should pass reference and finite-difference checks before UI adaptation.
- **Every update versus sampled diagnostics:** this release records every update and redraws live. Large widths, full-set gradient inspection, and many recorded steps cost time and memory. A worker and a bounded display cadence are sensible extensions if model sizes grow; preserve exact update counts and label any history downsampling.

## Hosting tradeoffs

| Option | Appropriate use | Tradeoff |
| --- | --- | --- |
| GitHub Pages + browser computation | This release; share a link with a class | Simple static hosting; speed depends on the student's device, no shared persistence |
| Download or clone | Offline exploration and modifying code | Student manages a local copy; opening generated HTML needs no Python |
| Shared CPU/GPU backend | Large image models or managed saved experiments | Requires compute funding, queues, per-user isolation, limits, and backend maintenance |

GitHub Pages publishes HTML/CSS/JavaScript; it does not run Python training. See [GitHub's Pages overview](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages) and [limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits). The Pages source for this project is `main` at `/`, with `.nojekyll`. Rebuild and commit both generated pages after editing sources. No cloud GPU or paid service is needed for the current model.

## Reusing the workflow for another architecture

1. Define the learning question and a task whose input/output can be understood visually. Keep a small analytic or numerical sanity example.
2. Specify comparable architectures, initialization, data ordering, optimizer, and parameter counts. Label any unmatched capacity or projection shortcuts.
3. Expose a clean forward trace: inputs, preactivations/activations, shortcut contributions, final output, and loss. Keep view selection separate from numerical state.
4. Validate derivatives against finite differences or an independent autograd implementation, including projections and shape changes.
5. Decide which projection is meaningful for hidden representations; display retained information and distinguish changing coordinates from changing features.
6. Add controlled interventions: preview one parameter without mutation, then provide a separate explicit training action. Check model and optimizer state restoration.
7. Compare fixed evaluation sets over time, log gradient magnitude and actual updates when relevant, and test more than one seed before generalizing.
8. Keep defaults navigable. Surface essential plots first, add optional views, and provide an experiment guide with reproducible settings.

Future CNN versions could trace feature channels, convolution kernels, activation gradients, and single-image classification loss. They require an explicit image-data/model source and a decision about browser versus server execution; those are outside this release.

## Verification and limitations

`tests/numerics.cjs` checks parameter counts, matched initialization, finite-difference derivatives including learned projections, and parameter/optimizer restoration after a sweep. `tests/browser_smoke.py` checks the generated page in isolated Chrome: mandatory/optional panels, defaults, heatmap legends, preview isolation, exact update-button counts, pause/reset, one-example steps, narrow layouts, and browser errors. It starts and stops its own temporary local server and browser.

These checks support the specified implementation. They do not prove convergence, certify all possible settings or browsers, or establish that residual networks always outperform plain ones. Browser storage, shareable model checkpoints, user uploads, automatic stopping, a GPU backend, and ImageNet are not implemented.
