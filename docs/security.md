# Credential handling / 凭据处理

Keep API keys, provider tokens, passwords, browser cookies, authentication databases and private keys outside the repository. Read provider credentials from environment variables; example configuration must contain empty values or explicit placeholders. Never attach populated `.env` files or raw authenticated request traces to issues or release archives.

密钥、令牌、密码、浏览器 Cookie 和私钥应存放在仓库之外，通过环境变量配置。示例配置只保留空值或明确的占位符；不要打包真实 `.env`、认证数据库或含鉴权头的原始请求日志。

## Local check

Install Gitleaks 8.30.1 or newer separately and put `gitleaks` on PATH. Run from the repository root:

```bash
uv run python scripts/check_secrets.py
uv run python scripts/check_secrets.py --history
```

The first command checks tracked files and non-ignored publication candidates. The second also checks every reachable Git revision. Both use the default Gitleaks detectors plus dedicated checks for MiMo Token Plan, LangSmith, AMap/Gaode and OpenAlex credential literals and authenticated URLs, inspect nested archives, and suppress matched values. The small allowlist, where present, is restricted to reviewed source hashes or code identifiers at exact paths; it does not exclude entire source or evidence directories. A passing scan records detector coverage, not a guarantee against every possible secret format.

第一条命令检查当前发布文件，第二条同时检查所有可达的 Git 历史。命中报告只记录路径、行号和规则，不输出密钥。代码检查脚本、摘要哈希和离线测试用的假令牌应与真实凭据区分。

## If a credential was committed

Revoke or rotate it at the provider first. Removing a value from the latest file does not remove earlier Git objects, existing clones, downloaded archives or provider-side access. A separately reviewed history rewrite may be required before public release. Do not test an exposed credential by making paid or authenticated provider calls.

已经提交过的凭据需要到服务商后台撤销或更换。修改当前文件不能清除旧提交、已有克隆或下载包；公开发布前还应确认历史是否完成清理。

## Service coverage / 非模型服务

The review also checks mapping/geocoding credentials (AMap/Gaode, including `securityJsCode`), tracing tokens (`LANGSMITH_API_KEY` / `LANGCHAIN_API_KEY`), scholarly API credentials (`OPENALEX_API_KEY` and URL parameters), and generic API-key/access-token query parameters. These belong in environment variables or untracked local configuration even when an SDK accepts a public browser key. An email address used to identify OpenAlex requests is not an API key.

检查范围包含高德地图／地理编码、LangSmith／LangChain 追踪、OpenAlex 文献接口及其他服务的 URL 鉴权参数。前端可见的地图 key 也按凭据处理；示例只保留空值或环境变量名。不要把概念图中的节点 key、缓存 key、token 计数或文件校验摘要当成服务密钥删除。

## Local environment / 本地环境文件

Copy `.env.example` to `.env`, fill supported variables locally, and keep `.env` untracked. The template has empty credential fields; tracing defaults to disabled. The optional service names document where credentials belong and do not imply that every repository implements those integrations.

```bash
cp .env.example .env
# Edit .env locally, then load it explicitly for the selected command:
uv run --env-file .env python scripts/check_secrets.py --history
```

Use `uv run --env-file .env python <entrypoint>` for an existing Python entrypoint, or use its documented environment loader. The scanner checks the publication payload and Git history; the deliberately ignored local `.env` is not a release artifact. Never copy secrets into the tracked template. `git check-ignore .env` should identify the ignore rule, while `git ls-files .env` should return no file.

真实 `.env` 仅保存在本地；`.env.example` 只提供变量名和空值。运行时显式使用 `--env-file .env`，即可向已有入口提供环境变量，不需要把密钥写回代码。已提交过的旧密钥仍需在相应服务商后台撤销或更换。
