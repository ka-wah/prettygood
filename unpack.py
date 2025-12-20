from pathlib import Path
import gzip
import shutil
import os


def ungzip_tree(root, *, delete_gz=True, overwrite=False, dry_run=False):
    root = Path(root)

    for gz_path in root.rglob("*.gz"):
        if not gz_path.is_file():
            continue

        out_path = gz_path.with_suffix("")  # removes only the final ".gz"

        if out_path.exists() and not overwrite:
            print(f"SKIP (exists): {gz_path} -> {out_path}")
            continue

        print(f"UNZIP: {gz_path} -> {out_path}")
        if dry_run:
            continue

        try:
            with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out, length=1024 * 1024)  # stream copy

            # optional: copy timestamps from .gz to output
            st = gz_path.stat()
            os.utime(out_path, (st.st_atime, st.st_mtime))

            if delete_gz:
                gz_path.unlink()

        except Exception as e:
            print(f"FAILED: {gz_path} ({e})")


if __name__ == "__main__":
    ungzip_tree(
        root=r"./results/dh_ret/all-all",
        delete_gz=True,   # set False if you want to keep the .gz too
        overwrite=False,  # set True if you want to replace existing outputs
        dry_run=False,    # set True to preview what would happen
    )
