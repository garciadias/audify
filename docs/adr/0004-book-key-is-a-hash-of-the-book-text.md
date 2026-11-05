# Book key is a hash of the book text

Re-publish has to find the Library item a book was published to before, so
that the folder, and with it every listener's progress, survives a re-run. We
identify a book by the **Book key**: a hash of its text. For an EPUB that is
the spine content, excluding the OPF metadata and the cover. For a PDF it is
the extracted text. Fixing a title or author in Calibre rewrites the EPUB but
leaves the text alone, so the key holds. A different edition, with different
text, correctly becomes a new book.

## Status

accepted

## Considered options

- **Whole-file hash.** Trivial to compute, but any metadata edit changes it.
  Re-publish would then miss the existing item and create a duplicate, the
  failure this design exists to stop.
- **`dc:identifier` (UUID or ISBN from the OPF), with the file hash as a
  fallback.** Calibre keeps the UUID across edits, but some EPUBs have no
  identifier or share junk ones, and PDFs have none at all.
- **Hash of the book text (chosen).** Stable under metadata edits, and the
  same for EPUB and PDF.

## Consequences

- The key is stored in every Item manifest (`.audify.json`). Changing how it
  is computed (normalisation, which spine items count) orphans every existing
  manifest, so the computation is versioned in the manifest.
- Text extraction must be deterministic for the same file. A parser upgrade
  that changes extracted text changes the key, which is another reason for the
  version field.
- Item identity is the Book key plus the rendition kind (task, translation
  target), so renditions and derived works of one book share a key without
  colliding.
