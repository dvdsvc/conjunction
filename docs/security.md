# Credential handling / 凭据处理

Keep API keys, provider tokens, passwords, browser cookies, authentication databases and private keys outside the repository. Read provider credentials from environment variables; example configuration must contain empty values or explicit placeholders. Never attach populated `.env` files or raw authenticated request traces to issues or release archives.

密钥、令牌、密码、浏览器 Cookie 和私钥应存放在仓库之外，通过环境变量配置。示例配置只保留空值或明确的占位符；不要打包真实 `.env`、认证数据库或含鉴权头的原始请求日志。

## Local check

Install Gitleaks 8.30.1 or newer separately and put `gitleaks` on PATH. Run from the repository root:

```bash
uv run python scripts/check_secrets.py
uv run python scripts/check_secrets.py --history
```

The first command checks tracked files and non-ignored publication candidates. The second also checks every reachable Git revision. Both use the default Gitleaks detectors plus the MiMo Token Plan credential pattern, inspect nested archives, and suppress matched values. The small allowlist, where present, is restricted to reviewed source hashes or code identifiers at exact paths; it does not exclude entire source or evidence directories. A passing scan records detector coverage, not a guarantee against every possible secret format.

第一条命令检查当前发布文件，第二条同时检查所有可达的 Git 历史。命中报告只记录路径、行号和规则，不输出密钥。代码检查脚本、摘要哈希和离线测试用的假令牌应与真实凭据区分。

## If a credential was committed

Revoke or rotate it at the provider first. Removing a value from the latest file does not remove earlier Git objects, existing clones, downloaded archives or provider-side access. A separately reviewed history rewrite may be required before public release. Do not test an exposed credential by making paid or authenticated provider calls.

已经提交过的凭据需要到服务商后台撤销或更换。修改当前文件不能清除旧提交、已有克隆或下载包；公开发布前还应确认历史是否完成清理。
