"""H263 controlled training-direction coverage with fixed features."""
import sys
kind,stage=sys.argv[1:]
assert kind in ['angle','isotropic']
if stage=='train':import run_bilin18_l17_response_supervision as protocol
elif stage=='evaluate':import run_bilin18_l17_residual_causal_transfer as protocol
elif stage=='analyze':import analyze_bilin18_l17_response_supervision as protocol
else:raise ValueError(stage)
protocol.STEM=protocol.ART/f'bilin18_l17_response_supervision_{kind}'
protocol.main()
