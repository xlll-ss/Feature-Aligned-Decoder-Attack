# Upload to GitHub

This folder is a local Git repository on branch `main`, without commits or
a remote. No files have been published. Use the GitHub website to create
an empty repository, then run these commands from this folder:

```bash
git status --short
git add .
git commit -m "Add FAD research code and experiment launchers"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your own. Choose repository visibility and an
appropriate project license before publishing. Dataset and model files are
excluded by `.gitignore`. Empty final summaries are not bundled as results.
The code still needs separately obtained datasets, pretrained models, and
the external IG implementation for the fair baseline experiment.
