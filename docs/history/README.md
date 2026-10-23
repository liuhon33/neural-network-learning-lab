# Publication notes and historical evidence

The [learning summary](../resnet-learning-summary.md) and [visualization playbook](../neural-network-visualization-playbook.md) were originally written on October 21, 2026 for Hongrui Liu's local ResNet experiment collection. Public editions were added on October 23, 2026 to preserve the questions, observations, and reusable workflow behind Neuro Network Visualizer.

“Example 2” and “Lab 02” refer to the hidden-signal and parameter-inspection lab from which this public project was developed. The other local experiments mentioned in the documents are historical context; their applications and model assets are not included here.

## What changed for publication

- Local application and source links now point to the public lab and repository.
- Setup/build instructions use the current project layout. The reusable prompt uses a reader-chosen directory and environment.
- The original folding interface is distinguished from the public workspace selector.
- The original questions, mathematical explanations, reported measurements, proposed extensions, and experimental reasoning are retained. The original local documents remain unchanged.

The documents are adapted editions, not verbatim archives. The [source manifest](source-manifest.json) records original relative filenames, SHA-256 hashes, destinations, and which files were adapted.

## Preserved evidence

| File | What it records |
| --- | --- |
| [Numerical checks](numerics.json) | Historical finite-difference, PyTorch, NumPy/PCA, optimizer, and parameter-preview comparisons. |
| [Inspector browser checks](inspector-verification.json) | The original lab's controls, layout, folding behavior, gradient sources, and preview checks. |
| [SGD browser comparison](function-sgd-verification.json) | Runs from the separate historical function-regression lab, referenced in the learning summary. |
| [Residual-correction screenshot](../images/history/correction-pca-wide.png) | Original feature/correction view at initialization. |
| [Parameter-inspection screenshot](../images/history/inspector-desktop-light.png) | Original parameter and gradient inspector. |

These reports and screenshots are copied byte-for-byte. Loopback URLs in report metadata identify the historical page tested; they are not public launch instructions. They are retained as evidence rather than rewritten to describe a different page.

The 36-block sensitivity measurements in the learning summary were reported during the original study; their specific checkpoint was not saved. These reports do not independently establish those sensitivity values. The fresh 500-update run illustrated in the public README has its own [capture record](../images/capture-info.json), and should not be conflated with the older run. Recorded historical numbers are not promises of identical results in every browser or version.

For current implementation and verification instructions, use the [technical guide](../architecture.md) and [developer README](../../README-legacy.md).
