# Generated outputs

`python reproduce.py` writes the public reproduction results here.

The generated files are ignored by Git. They can be deleted at any time with:

```bash
python reproduce.py --clean-only
```

After a successful run, `generated_files.txt` lists every generated file. Temporary staging and intermediate files are kept outside this directory and are removed automatically unless `--keep-work` is used.
