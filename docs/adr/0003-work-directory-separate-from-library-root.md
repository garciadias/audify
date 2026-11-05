# Work directory separate from the library root

Audify used to write every run into `data/output`. That directory is also a
Syncthing folder, mirrored to `/sdcard/Audiobooks` on mimix3, which is
Audiobookshelf's library root. As a result, scripts, `temp_NNN` folders and
`.tmp.m4b` parts were synced and scanned mid-run, and they showed up as bogus
books (for example `mlops_with_databricks/episodes`, 16 audio files and no
author). We decided to keep all working files in a work directory that is not
synced. A finished Library item is published into a configured library root
only after the run completes. It is assembled under a hidden temporary name and
then renamed into place, so neither Syncthing nor Audiobookshelf ever sees a
half-built book.

## Status

accepted

## Considered options

- **One directory plus a Syncthing `.stignore`** for working files. Smaller
  change, but it depends on ignore patterns that track every temp filename,
  and the library and the work tree stay entangled.
- **Write locally and let the user move items into the library.** No
  coupling, but a manual step on every book.
- **Separate work directory and atomic publish (chosen).** The library only
  ever receives complete Library items.

## Consequences

- The library root becomes configuration, not the output default. Existing
  folders under `data/output` (and the Syncthing mapping onto that path)
  need a one-off migration.
- The rename is atomic only within one filesystem, so the temporary name must
  sit inside the library root itself, not in the work directory.
- Atomic means atomic on this machine only. Audiobookshelf ignores any path
  with a dot-prefixed component (`fileUtils.shouldIgnoreFile`), so the hidden
  name hides the item while it is built. Syncthing, however, may carry the
  rename to mimix3 as files arriving one by one in the final folder. That is
  harmless: Audiobookshelf keys an item by its folder, so a late file updates
  the same book instead of creating a duplicate.
