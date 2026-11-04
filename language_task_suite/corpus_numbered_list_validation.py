"""Lightweight string and BPE checks for corpus numbered-list edits."""

def visible_bracket_answer(prefix):
    stack = []
    for char in prefix:
        if char in '([':
            stack.append(char)
        elif char in ')]':
            if not stack or (stack[-1], char) not in (
                    ('(', ')'), ('[', ']')):
                raise ValueError('Unbalanced visible bracket prefix')
            stack.pop()
    if not stack:
        raise ValueError('No visible opener for closer site')
    return ')' if stack[-1] == '(' else ']'


def _index(offsets, position):
    return next((i for i, (start, end) in enumerate(offsets)
                 if start <= position < end), None)


def aligned(doc, tokenizer, n_ctx):
    encoding = tokenizer.encode(doc.text, add_special_tokens=False)
    first = _index(encoding.offsets, doc.plan.markers[0][0])
    if first is None:
        return False
    indices = [first]
    for site in doc.sites:
        target = _index(encoding.offsets, site.start)
        source = _index(encoding.offsets, site.source_start)
        if (target is None or source is None or
                encoding.offsets[target][1] != site.end):
            return False
        indices.extend((target, source))
    return max(indices)-max(0, min(indices)-32) < n_ctx


def token_local_factorial(arms, tokenizer):
    """Require edits to alter only their containing BPE tokens to target."""
    encodings = {arm: tokenizer.encode(doc.text, add_special_tokens=False)
                 for arm, doc in arms.items()}
    base = arms[(0, 'round')]
    baseline = encodings[(0, 'round')]
    target_char = base.sites[-1].start
    limit = _index(baseline.offsets, target_char)
    if limit is None:
        return False
    for arm, doc in arms.items():
        enc = encodings[arm]
        if (_index(enc.offsets, target_char) != limit or
                enc.offsets[:limit+1] != baseline.offsets[:limit+1]):
            return False
        changed = [i for i, (a, b) in enumerate(zip(base.text, doc.text))
                   if a != b and i <= target_char]
        expected = {_index(baseline.offsets, pos) for pos in changed}
        if None in expected:
            return False
        actual = {i for i in range(limit+1)
                  if enc.ids[i] != baseline.ids[i]}
        if actual != expected:
            return False
    return True


def check(arms):
    base = arms[(0, 'round')]
    source = base.plan.source_text
    for (shift, style), doc in arms.items():
        assert len(doc.text) == len(source)
        changes = {i for i, (a, b) in enumerate(zip(source, doc.text))
                   if a != b}
        expected = ({start for start, _, _ in base.plan.markers}
                    if shift else set())
        if base.plan.bracket and style == 'square':
            expected.update(base.plan.bracket)
        assert changes == expected
        for site in doc.sites:
            assert doc.text[site.start:site.end] == site.answer
            if site.kind == 'bracket':
                assert visible_bracket_answer(doc.text[:site.start]) == \
                       site.answer
    assert arms[(0, 'round')].sites[-1].answer != \
           arms[(1, 'round')].sites[-1].answer
    if base.plan.bracket:
        assert arms[(0, 'round')].sites[0].answer != \
               arms[(0, 'square')].sites[0].answer
