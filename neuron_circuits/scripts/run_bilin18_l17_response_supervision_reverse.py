"""H262 reversed response-supervision split; unchanged fitting/evaluation."""
import sys
if sys.argv[1]=='train':
    import run_bilin18_l17_response_supervision as protocol
elif sys.argv[1]=='evaluate':
    import run_bilin18_l17_residual_causal_transfer as protocol
elif sys.argv[1]=='analyze':
    import analyze_bilin18_l17_response_supervision as protocol
else:raise ValueError('stage must be train, evaluate, or analyze')
protocol.STEM=protocol.ART/'bilin18_l17_response_supervision_reverse'
protocol.main()
