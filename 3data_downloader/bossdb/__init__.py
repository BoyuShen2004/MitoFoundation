"""BossDB stage-3 downloader package.

Public surface:
  metadata_client.discover()      → list[BossDBDataset]
  metadata_client.fetch_schema()  → dict
  inventory.pair_labeled_datasets() → list[dict]
  scriptgen.write_bossdb_outputs()  → (py_path, md_path)
  agent.main()                    → CLI entry-point
"""

