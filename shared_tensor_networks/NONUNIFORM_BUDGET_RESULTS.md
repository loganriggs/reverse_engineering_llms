# SVHN budget allocation: narrow leaves, wide upper layers

Three new CPUfits under the same standard-only SVHN protocol:68,257training/5,000validation,30epochs,modelseed200100,Adam.001,batch256,clip10,RMSNorm. Replace uniform-width budget trees with leafwidth16/8/4 atdepth2/3/4 andwidth32at everyhigherlevel. Choose leafwidth by nearestparametercount to thewidth32stack, keeping upperwidthfixed. No search over validation outcomes. One seed, fixedfinalepoch, officialtestunused.

This changes where parameters are allocated and changes initial functions; the varying-width model cannot generally share thewide stack's initialfunction. Fan-in initialization andseed are retained, but this isnot an exactlyfunction-matched causal intervention on representationwidth alone. Actualbudgets differ slightly, as listed.

| Depth | Stack parameters / val accuracy | Uniform budget tree parameters / val accuracy | New widths | New parameters / val accuracy | New train accuracy |
|---|---|---|---|---|---|
| 2 | 198976 / 84.86% | 197280 / 82.76% | [16, 32] | 197952 / 83.56% | 87.90% |
| 3 | 201024 / 83.94% | 197072 / 79.96% | [8, 32, 32] | 200000 / 78.26% | 83.30% |
| 4 | 203072 / 84.58% | 196872 / 65.78% | [4, 32, 32, 32] | 204096 / 77.98% | 81.25% |

Atdepth4the change improves validation by12.20percentagepoints, while remaining6.60points belowthe stack. It also improvestrainingfit substantially. Atdepth2thegain is.80points; atdepth3it loses1.70points. Thus uniformwidth budgeting was a consequential confound, but widerupperlayers are not a universalfix. The full experiment doesnotisolate sharing from all architecture/initialization/optimizer effects and is notseedreplicated.

## Validation

NonuniformTree inherits the tested bilinear node and traversal semantics while assigning widthperlevel. Six constantwidth cases reduce exactly to the originaltree inparameters, outputs andgradients. IndependentNumPyforward, finite-differenceinput/parametergradient checks, independentparametercounts and parameter-storageindependence pass. Allthree trainedmodels independentlyreplay training/validationmetrics andfinallogits, withtraining-onlychannelhistogramnormalization verified.

BriefCPU/CUDAchecks on thenewwidths passed withTF32disabled; maximum relativefloat32gradientdifference6.258e-6. This isbackendreadiness, notGPUtraining ortrajectoryequivalence. Checkpoints, sourceprotocols andcurves retained.

## Next

Close this allocationcontrol ratherthan treating onechosenwidthprofile asoptimal. Move tolarger verified SVHNextra-data training withstandard-onlyGPUcontrol andmatched-updateevaluation; extend depth2stage todepth3/4as planned. Retainallthreeparametercomparison types (equalwidth, uniformbudget, nonuniformbudget) as clearlydifferentconstraints; choose a boundedsubset forinitialGPUthroughput checks and reportits scope. Exactcircuitrecovery andproofsofscalability are notestablishedby classificationaccuracy.
