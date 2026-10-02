---
title: RagKno API
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# RagKno API

FastAPI backend for [RagKno](https://github.com/GitHpriyanshu23/Ragkno). The Docker image exposes the application on port `7860`. Configure all secrets in the Space **Settings**, attach persistent storage at `/data`, and keep `WEB_CONCURRENCY=1` for the in-process embedding and reranking models.
