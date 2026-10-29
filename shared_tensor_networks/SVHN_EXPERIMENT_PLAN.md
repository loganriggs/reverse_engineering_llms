# SVHN scaling plan — standard and allocation controls complete; extra data ready

Purpose: test whether shared-stack/untied-occurrence-tree findings transfer to a larger real-image dataset, and separate more standard examples from the easier additional-data distribution. This is predictive/optimization evidence, not circuit recovery.

Data source: Stanford official HTTP/HTTPS endpoints refused connections on this host. UCSD mirror https://cseweb.ucsd.edu/~weijian/static/datasets/svhn/ supplies identically named archives. Accept only exact reference checksums from torchvision's dataset implementation. Standard train archive passed MD5e26dedcc434d2e4c54c9b2d4a06d8373; SHA256435e94d69a87fde4fd4d7f3dd208dfc32cb6ae8af2240d066de1df7508d083b8. Extra archive is now verified: MD5a93ce644f1a588dc4d68dda5feec44a7, SHA256a133a4beb38a00fcdda90c9489e0c04f900b660ce8a316a5e854838379a71eb3. Bothtrain andextra memory-mapped arrays passed byte-equality checks; provenance is inresults/svhn_data.

prepare_svhn_cache.py writes uint8 NCHW memory-mapped arrays and int64 labels outside the repository, verifies full byte-array equality after writing, and records SHA256s with raw archive provenance. Digit10 maps to0; no other label remapping. Keep officialtest unused, and do not download it merely for development tuning.

Planned split: fixed seed200000 permutation of73,257 standard-training examples, reserve5,000 for validation, train on remaining68,257. Added-data regime appends531,131extra examples, yielding599,388training examples. Hold validation identical. Fit channel mean/std on standard-training68,257 only and freeze for bothregimes, avoiding a preprocessing change when addingextra. No augmentation or convolution in the first comparison; these are dense bilinear architectures.

Models: width32 RMSNorm stacks atdepth2/3/4, equal-width fullyuntied occurrence trees, nearest-budget narrower trees, twohidden64-wideReLUbaseline. Same levelwise matched-function initialization for stack/equalwidthtree. Record actual parameter counts and multiply-add estimates. Independent occurrence parameters remain independent despite equalinitialvalues.

Start standardtraining with fullpool,30epochs, Adam.001,batch256,clip10, fixedfinalepoch (fullcurves recorded). One fresh seed establishes runtime/optimization feasibility; follow with seedreplication, thenstandard+extra. Extra-data runs must report both examplesseen andoptimizerupdates: equalepochs confounds data size with compute. Retain matched-update checkpoints or a dedicated matched-update arm before making a sample-efficiency claim. No automatic claim of convergence at30epochs.

CPU-first. GenericCPUdispatch/pinnedcore has completed recent runs but prior nativecrash rootcause remains unresolved. Per-epoch model/optimizer/RNG-compatible checkpoints and terminalstatus needed. GPU remains untouched; only move largertraining workloads after checking availability against the existing project.

Validation: independently replay logits and CE/accuracy, verify imagechannel/layout conventions and data/split hashes, preserve failures. Use a fixed protocol before launch. Standard-only training is now complete and independently audited in SVHN_STANDARD_RESULTS.md. Nonuniform-width controls are now complete inNONUNIFORM_BUDGET_RESULTS.md. Extra-data training remains planned.

## GPU transition control

CPU/CUDAforward andgradient agreement checks passed fordepth2/4stack/tree in float32/64 withTF32disabled (results/vision_cuda/checks.json). A short numerical check isnottrajectoryequivalence forAdam. For an added-data GPUexperiment, include a standard-only GPUcontrol with the same precision/optimizer/backend so hardware isnotconfoundedwith data. Preserve the existing CPUstandard runs as separate evidence. GPUavailability must be rechecked beforelaunch. First paireddepth2runs can establish throughput before extending allplanneddepths; do not silently treatdepth2ascompletion of thedepth2–4plan.

## Budget-allocation control prompted by standard-data runs

Uniform-width budget matching shrinks the root representation along with every leaf. A separate nonuniform-width tree should keep upperlevelswidth32 and reduce only leafwidth to16/8/4 atdepth2/3/4. Withd3072,classcount10, parametercounts are197952/200000/204096 versusstacks198976/201024/203072. Counts follow sum_l[2 * 2^(D-1-l) * width_l * inputwidth_l] +10*width_(D-1), with inputwidth0=d and laterinputwidth=previouslevelwidth. This separate control is now implemented, correctness-tested and trained atall3depths; seeNONUNIFORM_BUDGET_RESULTS.md. It improvesdepth4substantially butnotallsettings. Do notfold it into the originaluniformwidthcomparison.

## Full-update control and replication decision (2026-10-29)

Depth-2/3 extra-data runs show better tree performance, including at8010updates, but the30epoch extra runs receive70260updates. Before attributing final differences to data, continue the independently audited depth3 standard models and Adam states from8010 to70260updates with the identical lr, clipping, batch size and epoch-specific shuffle sequence. This is263complete standard epochs plus39batches in epoch264. Retain the existing30epoch curve prefix; evaluate the prespecified final update without best-checkpoint selection. run_svhn_long_standard.py and audit_svhn_long_standard.py implement this bounded control. It is not an equal-FLOP stack/tree comparison; it equalizes updates across data regimes within each architecture.

The continuation is queued behind all depth4 audits to avoid GPU contention. Failures must remain visible. Do not change recipes based on validation outcomes mid-run. The standard/extra distribution difference remains even after this control.

Next replication should use five fresh initialization seeds with the split and data ordering fixed, retaining the original seed as development rather than silently pooling it as fresh confirmation. Target depth3 first. Include an additional approximately parameter-matched wider shared stack alongside width32 stack/tree; width changes initial functions, so preserve the equal-width pair separately. This tests whether additional tree capacity explains its data benefit. Report actual parameter counts and initialization output statistics. Choose the width by parameter count before examining replication outcomes. Neither five seeds on one split nor accuracy alone demonstrates general circuit recovery.
