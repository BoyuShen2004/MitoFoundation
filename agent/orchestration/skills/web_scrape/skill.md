---
id: web_scrape
title: Stage 1 site snapshot (scrape + probe)
label: Web scrape / probe
---
Use stage ``1web_scraper_01`` to fetch a landing page. **Workspace mode** (Pipeline Studio or ``agent.py URL --workspace``) writes ``1web_scraper_01/websites/<slug>/site.md`` and probe JSON under ``1web_scraper_01/outputs/<slug>.probe.json``. If the URL has no scheme, tooling defaults to ``http://`` (``https://`` is still allowed). If blocked, report HTTP status and stop.
