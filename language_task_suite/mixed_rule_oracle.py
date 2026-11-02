"""Exact observable next-token probabilities for the original Python sampler.

The forward filter marginalizes the latent bracket-start decision, rule, and
span length conditional on the visible prefix.
"""

from collections import defaultdict


def _add(out, token, probability):
    out[token] += probability


def core_distribution(gen, context):
    """Distribution of ``_sample_next_token(context)`` over token strings."""
    context = list(context)[-gen.window_size:]
    candidates = {}
    for cls, fn in [('trigram', gen._try_trigram),
                    ('bigram', gen._try_bigram),
                    ('skip_bigram', gen._try_skip_bigram),
                    ('induction', gen._try_induction)]:
        token, _ = fn(context)
        if token is not None:
            candidates[cls] = token
    if candidates and gen.mode == 'isolated':
        for cls in ('trigram', 'bigram', 'skip_bigram', 'induction'):
            if cls in candidates:
                return {candidates[cls]: 1.0}
    active = {cls: max(0.0, gen.mixing_weights[cls])
              for cls in candidates if cls in gen.mixing_weights}
    total = sum(active.values())
    if total > 0:
        out = defaultdict(float)
        for cls, weight in active.items():
            _add(out, candidates[cls], weight / total)
        return dict(out)

    # No active rule: the implementation first checks for a trigram trigger1.
    if context and gen.rules_by_class.get('trigram'):
        matching = [rule for rule in gen.rules_by_class['trigram']
                    if rule.trigger1 == context[-1]]
        if matching:
            out = defaultdict(float)
            for rule in matching:
                _add(out, rule.trigger2, 1.0 / len(matching))
            return dict(out)

    classes = [cls for cls in gen.mixing_weights
               if cls != 'noise' and gen.rules_by_class.get(cls)]
    weights = {cls: max(0.0, gen.mixing_weights[cls]) for cls in classes}
    weights['noise'] = max(0.0, gen.mixing_weights.get('noise', 0.01))
    z = sum(weights.values())
    if z <= 0:
        raise ValueError('No positive setup or noise weight')
    out = defaultdict(float)
    for cls, raw in weights.items():
        p = raw / z
        rules = gen.rules_by_class.get(cls, [])
        if cls == 'noise' or cls not in ('bigram', 'trigram', 'skip_bigram', 'induction'):
            # In the original implementation, choosing `bracket` here falls
            # through to the uniform-noise return. Bracket starts separately.
            for token in gen.vocab:
                _add(out, token, p / len(gen.vocab))
        elif cls == 'bigram':
            for rule in rules:
                _add(out, rule.trigger, p / len(rules))
        elif cls == 'trigram':
            for rule in rules:
                _add(out, rule.trigger1, p / len(rules))
        elif cls == 'skip_bigram':
            for rule in rules:
                window = context[-rule.max_skip:] if context else []
                token = rule.current_marker if rule.anchor in window else rule.anchor
                _add(out, token, p / len(rules))
        elif cls == 'induction':
            for rule in rules:
                _add(out, rule.trigger, p / len(rules))
    assert abs(sum(out.values()) - 1.0) < 1e-12
    return dict(out)


def outside_span_distribution(gen, context, bracket_eligible=True):
    """Path-conditional next-token distribution outside an inserted bracket span."""
    core = core_distribution(gen, context)
    p = gen.mixing_weights.get('bracket', 0.0) if (
        bracket_eligible and gen.rules_by_class.get('bracket')) else 0.0
    if p < 0 or p > 1:
        raise ValueError('Bracket-start probability outside [0,1]')
    out = defaultdict(float)
    for token, mass in core.items():
        _add(out, token, (1-p)*mass)
    if p:
        for rule in gen.rules_by_class['bracket']:
            _add(out, rule.open_token, p/len(gen.rules_by_class['bracket']))
    assert abs(sum(out.values()) - 1.0) < 1e-12
    return dict(out)


def _transitions(gen, prefix, state, total_length):
    """Joint distribution of next observed token and next hidden span state."""
    t = len(prefix)
    out = defaultdict(float)
    if t == 0:
        # sample_sequence seeds each stream with one uniform noise token.
        for token in gen.vocab:
            out[(token, None)] += 1/len(gen.vocab)
        return out
    if state is None:
        eligible = bool(gen.rules_by_class.get('bracket')) and t+3 <= total_length
        p_start = gen.mixing_weights.get('bracket', 0.0) if eligible else 0.0
        for token, p in core_distribution(gen, prefix).items():
            out[(token, None)] += (1-p_start)*p
        if eligible and p_start:
            rules = gen.rules_by_class['bracket']
            for i, rule in enumerate(rules):
                for remaining in range(1, rule.max_content_len+1):
                    out[(rule.open_token, (i, remaining))] += (
                        p_start/len(rules)/rule.max_content_len)
        return out
    i, remaining = state
    rule = gen.rules_by_class['bracket'][i]
    if remaining:
        for token in rule.content_tokens:
            out[(token, (i, remaining-1))] += 1/len(rule.content_tokens)
    else:
        out[(rule.close_token, None)] = 1.0
    return out


def observable_distribution(gen, prefix, total_length):
    """Exact P(next token | observed prefix) for LanguageGenerator.sample_sequence.

    This forward filter marginalizes whether an observed opener came from the
    bracket-start branch or ordinary core/noise emission, and the unobserved
    sampled span length. ``total_length`` is the intended complete sequence
    length, because it controls whether a new span may start near the end.
    """
    prefix = list(prefix)
    if len(prefix) >= total_length:
        raise ValueError('No next token in a completed sequence')
    states = {None: 1.0}
    for t, token in enumerate(prefix):
        updated = defaultdict(float)
        for state, mass in states.items():
            for (candidate, next_state), p in _transitions(
                    gen, prefix[:t], state, total_length).items():
                if candidate == token:
                    updated[next_state] += mass*p
        normalizer = sum(updated.values())
        if normalizer <= 0:
            raise ValueError(f'Impossible prefix at position {t}: {prefix!r}')
        states = {state: mass/normalizer for state, mass in updated.items()}
    result = defaultdict(float)
    for state, mass in states.items():
        for (token, _), p in _transitions(gen, prefix, state,
                                          total_length).items():
            result[token] += mass*p
    assert abs(sum(result.values()) - 1.0) < 1e-12
    return dict(result)


def observable_distributions_for_sequence(gen, tokens, total_length=None):
    """All prefix-conditioned distributions in one forward-filter pass."""
    tokens = list(tokens)
    total_length = len(tokens) if total_length is None else total_length
    if len(tokens) > total_length:
        raise ValueError('Observed sequence exceeds total_length')
    states = {None: 1.0}
    distributions = []
    for t, observed in enumerate(tokens):
        joint = defaultdict(float)
        for state, mass in states.items():
            for (token, next_state), p in _transitions(
                    gen, tokens[:t], state, total_length).items():
                joint[(token, next_state)] += mass*p
        distribution = defaultdict(float)
        for (token, _), p in joint.items():
            distribution[token] += p
        distributions.append(dict(distribution))
        posterior = defaultdict(float)
        for (token, next_state), p in joint.items():
            if token == observed:
                posterior[next_state] += p
        normalizer = sum(posterior.values())
        if normalizer <= 0:
            raise ValueError(f'Impossible observed token {observed!r} at {t}')
        states = {state: p/normalizer for state, p in posterior.items()}
    return distributions
