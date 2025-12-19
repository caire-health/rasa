## Summary

Upgrade TensorFlow to 2.15.0 to address high-security Keras vulnerability CVE-2024-3660 (Deserialization of Untrusted Data).

## Changes

- Upgrade TensorFlow (all platform variants) from 2.12.1 → 2.15.0
- Upgrade tensorflow-text to 2.15.0
- Update Keras import paths for 2.15 compatibility (`tensorflow.python.keras.utils`)
- Pin jax/jaxlib to 0.4.30 (must match exactly to avoid runtime errors)
- Bump typing-extensions to ^4.12

## Security Notes

| CVE | Severity | Status |
|-----|----------|--------|
| CVE-2024-3660 | High | ✅ Fixed by this upgrade |
| CVE-2025-12060 | High | ⚠️ Risk accepted - `keras.utils.get_file()` not used by caire-services |
| CVE-2025-12058 | Medium | ⚠️ Risk accepted - `keras.Model.load_model` with `StringLookup` not used |

## Why pin jax/jaxlib?

These packages must be the exact same version to work together, but Poetry doesn't enforce this constraint. Without pinning, Poetry can resolve to mismatched versions (e.g., jax 0.4.13 + jaxlib 0.4.30) causing runtime errors:
```
RuntimeError: jaxlib version 0.4.30 is newer than and incompatible with jax version 0.4.13
```

## Test Plan

- [ ] caire-services CI passes (`rasa train`, `rasa data validate`)
- [ ] Verify NLU pipeline works with existing training data

## After Merge

Update caire-services to use the new rasa version:
```bash
cd services/rasa-server
poetry update rasa
git add poetry.lock
git commit -m "chore: update rasa fork to TF 2.15 (security fix)"
```
