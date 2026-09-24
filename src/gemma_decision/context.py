# SPDX-License-Identifier: Apache-2.0
"""Fixed native text context tiers; no positional extrapolation."""
MAX_REQUEST_TOKENS = 524288


def context_settings(max_input_tokens=None, media=False):
    limit = (8192 if media else 65535) if max_input_tokens is None else max_input_tokens
    ceiling = 8192 if media else 262143
    if type(limit) is not int or not 1 <= limit <= ceiling:
        raise ValueError(f'Input token limit must be an integer in 1..{ceiling}')
    if media:
        return limit, 16384, 3 * 1024**3
    for context, gib in [(65536, 3), (131072, 4), (262144, 8)]:
        if limit < context:
            return limit, context, gib * 1024**3
