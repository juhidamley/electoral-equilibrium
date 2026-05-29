# data/archives/

Raw archive data used for constructing the fine-tuning dataset.
This directory lives only on the Intel Mac (primary archive machine).
Other machines receive processed outputs via Syncthing, never raw archives.

## Subdirectories

| Directory | Contents | Source |
|---|---|---|
| `news/3dlnews/` | 3DLNews2 URL metadata list (filtered to shock windows) | HPC fetch-and-parse |
| `news/webhose/` | Webhose news dumps | Webhose API |
| `reddit/` | Sampled Reddit posts per subreddit | collectors/reddit.py |
| `discord/` | Discord archive exports (if available) | Manual export |

## Fetch-and-parse workflow (3DLNews2)

Never store 8M raw HTML pages locally. Instead:
1. Download URL metadata list to `news/3dlnews/` (URLs + dates + outlet names)
2. Filter to shock date windows and target outlets before fetching any HTML
3. Fetch-and-parse on HPC scratch via SLURM (scripts/score_array.sh pattern)
4. rsync resulting `articles.jsonl` files back to Intel Mac
5. Raw HTML is never written to disk

## Ownership

Only the Intel Mac writes to `rawdata/social/` and `rawdata/articles/`.
Never overwrite another machine's files. See CLAUDE.md §Collectors for the
full write-ownership rules.
