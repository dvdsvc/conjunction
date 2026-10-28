# Environment configuration / 环境配置

The documented graph construction, training and evaluation workflows use local data and do not require API credentials. Keep credentials for any separate data-acquisition workflow in local environment variables.

已提供的图构造、训练和评估使用本地数据，无需 API 密钥。其他数据获取流程如需凭据，可通过本地环境变量配置。

## Local setup

```bash
cp .env.example .env
```

Fill in the supported variables in your local `.env`. The file is ignored by Git; keep the committed template's credential fields empty. For entrypoints that read process variables, add `--env-file .env` to the documented `uv run` command. See the repository's running guide for provider and data options.

在本地 `.env` 中填写所需变量，提交的示例文件保持空白。读取进程环境变量的入口可通过 `uv run --env-file .env ...` 载入配置，具体运行方式见对应说明。

## Repository check

With [Gitleaks](https://github.com/gitleaks/gitleaks) installed and on PATH:

```bash
uv run python scripts/check_secrets.py --history
```

This checks publication files and reachable Git history using the repository's detector configuration. Reports show locations without credential values.
