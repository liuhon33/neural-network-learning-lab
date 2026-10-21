# Neural Network Learning Lab

**Watch a neural network learn, then inspect what changed.** Compare a plain dense network and a residual network on the same two-input, one-output function. Move through hidden layers, inspect individual weights and gradients, and explore the loss around a selected parameter.

**[Open the interactive lab](https://liuhon33.github.io/neural-network-learning-lab/)** · **[How to use it](https://liuhon33.github.io/neural-network-learning-lab/guide.html)** · **[Technical guide](docs/architecture.md)**

No account, Python installation, GPU, or server setup is needed to play. All training runs in your browser. The page loads no third-party scripts, sends no training data to a service, and stores no model state after a reload.

## Start exploring

1. Open the lab. It starts with **36 hidden blocks, 10 features per layer, seed 7**, and 1,024 training examples.
2. Move the block slider and compare the hidden features and gradients before training.
3. Choose **Train 500 updates**. Pause to inspect a fixed model; choose another training button to continue.
4. Enable **Weights and gradients** or **One-weight experiment** for a closer look. Display choices do not reset the networks.

The [separate student guide](https://liuhon33.github.io/neural-network-learning-lab/guide.html) explains every control, what each plot measures, and experiments to try.

## What you can see

- Always visible: input and hidden-feature maps, target and final prediction surfaces, loss history, and gradient magnitude across layers.
- Optional: the added residual correction, exact layer signals, signed parameter/gradient heatmaps with numeric color bars, and a single-example loss slice through one selected weight.
- Adjustable depth (3/12/24/36), shared or individual widths (1–32), seed, and training-set size. Architecture/data changes reset both models.
- One, 100, 500, or 1,000 additional updates, with progress and pause. Training uses matching shuffled batches and Adam at 0.003.
- Live shared PCA for hidden features wider than two dimensions, synchronized surface rotation, and explicit one-example training.

This release contains the dense function-regression inspection lab. It does not include ImageNet, CNNs, pretrained models, or the separate optimizer-comparison experiment. Residual connections are useful to study, but are not a guarantee that every run succeeds. This is an educational experiment, not a benchmark claim.

## Run or modify it locally

Download the repository and open `index.html` in a modern browser. It is self-contained and also works offline. Alternatively:

```sh
git clone https://github.com/liuhon33/neural-network-learning-lab.git
cd neural-network-learning-lab
python -m http.server 8000
```

Open `http://localhost:8000`. Python is needed only for this optional local server or rebuilding the files.

Edit the files in `src/`, then regenerate the committed standalone pages:

```sh
python scripts/build.py
node tests/numerics.cjs
```

The builder needs only Python's standard library. Numerical tests need Node.js. Optional browser checks use Chrome/Chromium and Python's `websocket-client` package:

```sh
python -m pip install websocket-client
python tests/browser_smoke.py
```

Set `CHROME_PATH` if Chrome is not found automatically. Screenshots and test results go into ignored `verification/`. See [verification and limitations](docs/architecture.md#verification-and-limitations).

## Project map

| Path | Purpose |
| --- | --- |
| `index.html`, `guide.html` | Generated, ready-to-host pages |
| `src/dense-engine.js` | Forward pass, analytic backpropagation, Adam, parameter sweeps, PCA |
| `src/inspector.js` | Gradient sources, history, heatmaps, loss slices |
| `src/lab.js`, `src/lab.html`, `src/lab.css` | Interactive experiment and original plot layout |
| `src/site.css`, `src/guide.html` | Public website styling and student instructions |
| `scripts/build.py` | Reproducible standalone-page builder |
| `tests/` | Numerical and browser verification |
| `docs/architecture.md` | Equations, design choices, hosting, extension workflow |

## Hosting and contributions

The public site is served by **GitHub Pages from `main`, repository root**. Committing rebuilt pages updates the site. A fork can enable the same setup in Settings → Pages. Change the repository links in `scripts/build.py`, the guide, and README for your fork.

For this small lab, static hosting is sufficient: each visitor supplies the computation through their browser. A backend becomes relevant for larger image models, shared saved experiments, or GPU training; see the [hosting tradeoffs](docs/architecture.md#hosting-tradeoffs).

Bug reports and focused improvements are welcome. Include architecture, widths, seed, sample count, update count, selected input, gradient source, and browser when reporting a numerical or display issue. Changes to the learning equations should include a numerical check and update the technical guide.

Created by Hongrui Liu from an interactive learning project developed with AI coding assistance. Released under the [MIT License](LICENSE).
