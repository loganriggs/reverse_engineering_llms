# SVHN scaling protocol draft — not yet training

Purpose: test whether shared-stack/untied-occurrence-tree findings transfer to a larger real-image dataset, and separate more standard examples from the easier additional-data distribution. This is predictive/optimization evidence, not circuit recovery.

Data source: Stanford official HTTP/HTTPS endpoints refused connections on this host. UCSD mirror https://cseweb.ucsd.edu/~weijian/static/datasets/svhn/ supplies identically named archives. Accept only exact reference checksums from torchvision's dataset implementation. Standard train archive passed MD5e26dedcc434d2e4c54c9b2d4a06d8373; SHA256435e94d69a87fde4fd4d7f3dd208dfc32cb6ae8af2240d066de1df7508d083b8. Extra archive is still downloading; no verification or availability claim until complete.

prepare_svhn_cache.py writes uint8 NCHW memory-mapped arrays and int64 labels outside the repository, verifies full byte-array equality after writing, and records SHA256s with raw archive provenance. Digit10 maps to0; no other label remapping. Keep officialtest unused, and do not download it merely for development tuning.

Planned split: fixed seed200000 permutation of73,257 standard-training examples, reserve5,000 for validation, train on remaining68,257. Added-data regime appends531,131extra examples, yielding599,388training examples. Hold validation identical. Fit channel mean/std on standard-training68,257 only and freeze for bothregimes, avoiding a preprocessing change when addingextra. No augmentation or convolution in the first comparison; these are dense bilinear architectures.

Models: width32 RMSNorm stacks atdepth2/3/4, equal-width fullyuntied occurrence trees, nearest-budget narrower trees, twohidden64-wideReLUbaseline. Same levelwise matched-function initialization for stack/equalwidthtree. Record actual parameter counts and multiply-add estimates. Independent occurrence parameters remain independent despite equalinitialvalues.

Start standardtraining with fullpool,30epochs, Adam.001,batch256,clip10, fixedfinalepoch (fullcurves recorded). One fresh seed establishes runtime/optimization feasibility; follow with seedreplication, thenstandard+extra. Extra-data runs must report both examplesseen andoptimizerupdates: equalepochs confounds data size with compute. Retain matched-update checkpoints or a dedicated matched-update arm before making a sample-efficiency claim. No automatic claim of convergence at30epochs.

CPU-first. GenericCPUdispatch/pinnedcore has completed recent runs but prior nativecrash rootcause remains unresolved. Per-epoch model/optimizer/RNG-compatible checkpoints and terminalstatus needed. GPU remains untouched; only move largertraining workloads after checking availability against the existing project.

Validation: independently replay logits and CE/accuracy, verify imagechannel/layout conventions and data/split hashes, preserve failures. Use a fixed protocol before launch. This document is a plan, not evidence that SVHNtraining has run.
