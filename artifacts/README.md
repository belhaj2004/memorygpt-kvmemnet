# Optional trained artifacts

The app runs immediately in portfolio demo mode. To activate the original trained network, place these files here:

- `kvmemnet_model_final.pt`
- `data.pkl`
- `vocab.pkl`

The notebook originally saved the database as `data`, the vocabulary as `vocab`, and the weights as `kvmemnet_model_final.pt` in Google Drive. Rename the pickle files as shown above.

Large artifacts should be stored with Git LFS, a release asset, or downloaded at startup rather than committed directly to a normal Git repository.
