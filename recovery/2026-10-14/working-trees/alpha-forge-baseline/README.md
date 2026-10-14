# alpha-forge-baseline working-tree recovery

Repository: `https://github.com/cdayAI/Project-Freedom-.git`. Exact base commit: `d57d7299366c92bcabbfd106402179e1945426a0`.

This package preserves selected authored source/configuration/documentation changes and safe untracked files. Runtime records, private data, ignored files and any exclusions listed in `manifest.json` remain in the original local checkout. This package alone does not make the original workspace safe to delete.

Restore in a separate clone at the exact base commit, never directly onto current main. Obtain the base commit from the repository's branches/PR refs (or the accompanying unpublished-history bundle when required). Set `core.autocrlf=false` for a normalized checkout, then:

1. Check out `d57d7299366c92bcabbfd106402179e1945426a0` on a new recovery branch.
2. If `staged-source.patch` is nonempty, run `git apply --check <path-to-staged-source.patch>` and `git apply --index <path-to-staged-source.patch>`.
3. If `unstaged-source.patch` is nonempty, run `git apply --check <path-to-unstaged-source.patch>` and `git apply <path-to-unstaged-source.patch>`.
4. Copy the contents of `untracked-source/` into the recovery checkout, preserving relative paths, if present.
5. Compare each selected file with its `normalized_git_blob` using `git hash-object --path=<relative-path> <file>`. Raw SHA-256 values also preserve the original working-copy bytes, which may differ only in line endings after patch application.

Restoration was checked against the pinned base: patches apply in order and all selected normalized contents match. This was a file-recovery test, not an application, clinical, financial or gameplay validation. No broker calls, application execution, original checkout mutations or remote writes were performed for this export.
