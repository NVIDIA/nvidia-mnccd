# nvidia-mnccd

Nvidia Multi-Node Confidential Compute Daemon (nvidia-mnccd) is an open source privileged usermode daemon that is supposed to run either on baremental or inside a confidential VM (CVM). It is responsible for key/IV management across nodes, multi-node discovery, multi-node attestation and ensuring that all the nodes are assigned to the same tenant.

## Building nvidia-mnccd

The daemon is a Rust crate built with [Cargo](https://doc.rust-lang.org/cargo/). There is a single build that always includes gRPC TLS support: **TLS-PSK** (default at runtime, via OpenSSL and `tonic-tls`), **mutual TLS** (rustls with operator-supplied PEM certificates), and plain HTTP/2 when disabled. HKDF key derivation for NVLE flows uses the `hkdf` crate. Whether gRPC connections use TLS, and which mode, is chosen at **runtime** via `[tls].mode` in config and the `--no-tls` flag (see [Running nvidia-mnccd](#running-nvidia-mnccd)).

**Prerequisites**

- A recent stable Rust toolchain (`rustup` recommended).
- **OpenSSL development libraries** on the build host (for example `libssl-dev` on Debian/Ubuntu, `openssl-devel` on RHEL). Required for the default PSK transport (`openssl-sys` / `tonic-tls`).
- **`nvml.h` from CUDA 13.5 or newer.** `build.rs` generates NVML bindings from this header (only the header is needed at compile time; `libnvidia-ml.so` is loaded at runtime). Older toolkits lack the NVLE/remap-table APIs and the build fails.

**`nvml.h` location**

The header is resolved in this order:

1. **`NVML_HEADER_DIR`** — directory that contains `nvml.h` (not the file itself). Use this when the CUDA toolkit is not in a default path, or when `/usr/local/cuda` does not point at 13.5+.
2. Otherwise the first readable file among:
   - `/usr/include/nvml.h`
   - `/usr/local/include/nvml.h`
   - `/usr/local/cuda/include/nvml.h` (typical CUDA SDK install; `/usr/local/cuda` is usually a symlink to the active toolkit, e.g. `/usr/local/cuda-13.5/include/nvml.h`)

A versioned toolkit that is not linked as `/usr/local/cuda` is not searched automatically:

```sh
# Example: CUDA 13.5 installed at /usr/local/cuda-13.5
NVML_HEADER_DIR=/usr/local/cuda-13.5/include cargo build
```

**Debug builds**

```sh
cargo build
```

**Release builds**

```sh
cargo build --release
```

The compiled binary is `target/debug/nvidia-mnccd` or `target/release/nvidia-mnccd`, depending on the profile.

## Running nvidia-mnccd

**Configuration directory**

At startup the process loads `mnccd_config.toml` from a directory chosen in this order:

1. **`MNCCD_DATA_DIR`** (runtime override).
2. **`CARGO_MANIFEST_DIR`** (set automatically when you use `cargo run`).
3. Compile-time default `/etc/nvidia-mnccd` (packagers can override at build time with `MNCCD_DEFAULT_DATA_DIR=<path> cargo build`).

Example layout:

```text
$MNCCD_DATA_DIR/
  mnccd_config.toml
  tls/                   # only required when [tls].mode = "mtls"
    ca-root.crt          # cluster CA certificate (trust anchor)
    server.key / server.crt
    client.key / client.crt
```

**TLS modes**

Set `[tls].mode` in `mnccd_config.toml` (all nodes must use the same mode):

| Mode | Config value | Description |
| --- | --- | --- |
| **TLS-PSK** (default) | `psk` | OpenSSL TLS 1.2 DHE-PSK (forward secrecy). No certificate files under `tls/`. Same `psk_identity` and `psk_key` on every node. |
| Mutual TLS | `mtls` | rustls with PEM certificates. Requires operator-supplied files under `tls/` on every node (see below). |
| Plaintext | `none` | Unencrypted gRPC. Usually selected with `--no-tls` on the CLI (overrides config). |

If the `[tls]` section is omitted, MNCCD defaults to **`psk`**, but **`psk_identity` and `psk_key` are required** and must be set explicitly (startup fails if either is missing or empty).

**TLS-PSK (`psk`, default)**

PSK encrypts the gRPC mesh using a shared cluster secret. Each node is both TLS client and server; peers authenticate with the same pre-shared identity and key. Ciphersuites are **DHE-PSK-AES256-GCM-SHA384** and **DHE-PSK-AES128-GCM-SHA256**: ephemeral Diffie-Hellman is mixed with the PSK so session keys have forward secrecy—captured traffic cannot be decrypted later from the cluster PSK alone.

| Field | Meaning |
| --- | --- |
| `[tls].mode` | `"psk"` (default if `[tls]` is omitted). |
| `[tls].psk_identity` | PSK identity sent in the TLS handshake (public label, like a username). **Required** when mode is `psk`. Must match on every node. |
| `[tls].psk_key` | Pre-shared key bytes (the secret). **Required** when mode is `psk`. Must match on every node. |

Limits (OpenSSL): identity ≤ ~127 bytes; key ≤ 256 bytes.

Example:

```toml
[tls]
mode = "psk"
psk_identity = "mnccd-cluster"
psk_key = "replace-with-a-long-random-secret"
```

Provision the same `psk_identity` and `psk_key` on every node (ConfigMap, Secret, or baked into `mnccd_config.toml`). The sample placeholder key in the shipped `mnccd_config.toml` (`replace-with-a-long-random-secret`) is **rejected at startup**—replace it before running.

**Mutual TLS (`mtls`)**

TLS material uses the **same directory** as `mnccd_config.toml` (see resolution order above). With `[tls].mode = "mtls"`, install the following PEM files under `tls/` on every node **before** starting the daemon. MNCCD does **not** generate certificates or keys.

| File | Role |
| --- | --- |
| `tls/ca-root.crt` | Cluster CA certificate (trust anchor). Must be the same on every node. |
| `tls/server.crt` / `tls/server.key` | This node's gRPC server identity. |
| `tls/client.crt` / `tls/client.key` | This node's gRPC client identity when dialing peers. |

Server certificates must present a DNS SAN (or name) matching `[tls].server_name` (default `mnccd`) — peer clients verify against that domain. Private keys should be mode `0600`.

`tls/ca-root.key` is **not** read by MNCCD at runtime; keep the CA private key offline and use it only when issuing node certificates.

**Optional mTLS peer certificate checks**

With `mtls`, two optional `[tls]` keys restrict which certificates signed by `tls/ca-root.crt` a node accepts from its peers. They apply on top of the `[cluster].node_ips` allowlist and are useful when one CA issues certificates for more than one MNCCD cluster. When omitted or empty, behavior is unchanged. Use the same values on every node.

| Field | Meaning |
| --- | --- |
| `[tls].server_name` | DNS name that every peer's server certificate must present as a DNS SAN; clients verify it during the TLS handshake. Default `mnccd`. IP addresses and wildcards are rejected. |
| `[tls].allowed_client_organizations` | Subject Organization (O) values accepted on peer client certificates. The server denies requests whose client certificate has none of them (exact, case-sensitive match). Default: no check. |

Setting either key while `[tls].mode` is not `mtls` is a startup error. `--no-tls` turns them off along with TLS.

```toml
[tls]
mode = "mtls"
server_name = "group-a.example.com"
allowed_client_organizations = ["example-group-a"]
```

The TOML file defines the gRPC listen port and the cluster membership:

| Field | Meaning |
| --- | --- |
| `[server].server_port` | TCP port the gRPC server binds on (each node uses the same port number). |
| `[cluster].node_ips` | List of all node IP addresses in the cluster. The **leader** is the node with the numerically smallest IP; that node runs NVLE setup when not skipped (see below). |
| `[retry_policy].max_retries` | Maximum number of retries for transient gRPC failures (for example `Unavailable`). |
| `[retry_policy].initial_backoff_ms` | Base delay in milliseconds before the first retry; each subsequent retry doubles the delay until capped by `max_backoff_ms`. |
| `[retry_policy].max_backoff_ms` | Maximum delay between retries, in milliseconds (exponential backoff is capped at this value). |
| `[tls].mode` | gRPC transport: `psk` (default), `mtls`, or `none`. All nodes must match. |
| `[tls].psk_identity` | TLS-PSK identity. **Required** when mode is `psk`; must match on every node. |
| `[tls].psk_key` | TLS-PSK secret. **Required** when mode is `psk`; must match on every node. |
| `[tls].server_name` | Optional, `mtls` only. DNS SAN required on peer server certificates (default `mnccd`). |
| `[tls].allowed_client_organizations` | Optional, `mtls` only. Subject Organization values accepted on peer client certificates (default: no check). |

Example `mnccd_config.toml`:

```toml
[server]
server_port = 50051

[cluster]
node_ips = ["10.0.0.1", "10.0.0.2"]

[retry_policy]
max_retries = 10
initial_backoff_ms = 100
max_backoff_ms = 100000

[tls]
mode = "psk"
psk_identity = "mnccd-cluster"
psk_key = "replace-with-a-long-random-secret"
```

Example with **mutual TLS** instead:

```toml
[tls]
mode = "mtls"
```

Then place `ca-root.crt`, `server.{crt,key}`, and `client.{crt,key}` under `tls/` as described above.
**Command-line arguments**

Run `nvidia-mnccd --help` (or `-h`) for usage, including where `mnccd_config.toml` is loaded from. Run `nvidia-mnccd --version` (or `-V`) to print the crate version.

| Argument | Required | Description |
| --- | --- | --- |
| `--skip-nvle` | No | When set, the leader node **does not** call `setup_nvle_on_all_gpus()` after peer echo checks. Omit for the normal path where the leader configures NVLE on GPUs across the cluster. |
| `--no-tls` | No | Force plaintext gRPC (`[tls].mode = "none"`), regardless of config. Must be used on **all** nodes together, or on **none** of them. Strictly for development platforms only; not recommended on production. |
| `--daemonize` | No | Fork into the background and redirect stdout/stderr to `/tmp/nvidia-mnccd.out` and `/tmp/nvidia-mnccd.err`. For manual runs without systemd. **Do not** use with the systemd unit (see below). |
| `--log-level <LEVEL>` | No | Base log verbosity: `error`, `warn`, `info` (default), `debug`, or `trace`. Overridden by the `RUST_LOG` environment variable when set. See [Logging](#logging). |

**TLS mode (all nodes must match)**

Every node must use the same transport: **PSK** (default), **mTLS**, or **plaintext** (`--no-tls` / `tls.mode = "none"`). Mixing modes (for example PSK on one node and `--no-tls` on another) is not supported and peer connectivity will fail.

Examples:

```sh
# Usage summary
/path/to/nvidia-mnccd --help

# From the repo (Cargo sets CARGO_MANIFEST_DIR; uses ./mnccd_config.toml)
# Default: TLS-PSK when [tls].mode is omitted; psk_identity and psk_key are still required
cargo run

# Installed or copied binary: set config directory
export MNCCD_DATA_DIR=/path/to/config/dir
/path/to/nvidia-mnccd

# Skip NVLE setup on the leader
/path/to/nvidia-mnccd --skip-nvle

# Plain gRPC (no TLS)—must be set on every node in the cluster
/path/to/nvidia-mnccd --no-tls

# Manual background run (logs under /tmp; not for systemd)
/path/to/nvidia-mnccd --daemonize
```

### Privileges

| | root | non-root |
| --- | --- | --- |
| Foreground (default) | Supported | **Not supported** — MNCCD exits with an error |
| `--daemonize` | Supported | Supported |

GPU and other privileged operations may still fail when MNCCD is not running as root.

## Production deployment

Production nodes should run MNCCD under **systemd**. The binary stays in the foreground from systemd’s perspective (`Type=simple`); systemd supervises the process and collects logs. Do **not** pass `--daemonize` in the unit file.

### Run package

Build release binaries into `bin/release/<arch>/` (with matching `nvidia-mnccd.md5sum` and `nvidia-mnccd.sha256sum` files), then create a tarball:

```sh
./scripts/generate_run_package.sh --arch "$(uname -m)"
```

The tarball under `_out/` contains:

- `nvidia-mnccd` and checksum sidecars
- `mnccd_config.toml`, `LICENSE`, `third-party-notices.txt`
- `mnccd_run_package_installer.sh`
- `nvidia-mnccd.service`

TLS keys and certificates are **not** included in the package; provision them separately when using mTLS.

On each node, extract the tarball and run the installer as root:

```sh
tar -xzf nvidia-mnccd-run-*.tar.gz
sudo ./mnccd_run_package_installer.sh
```

The installer copies:

| Artifact | Install path |
| --- | --- |
| Binary | `/usr/bin/nvidia-mnccd` |
| Config | `/etc/nvidia-mnccd/mnccd_config.toml` (mode `0600`; contains `psk_key` when using TLS-PSK) |
| TLS directory | `/etc/nvidia-mnccd/tls/` (created empty; used for **mTLS** only) |
| LICENSE / third-party-notices.txt | `/usr/share/nvidia/mnccd/doc/` |
| systemd unit | `/usr/lib/systemd/system/nvidia-mnccd.service` |

Edit `/etc/nvidia-mnccd/mnccd_config.toml` for the cluster before starting the service. The bundled unit uses **TLS-PSK** when `[tls].mode` is `psk` or omitted; set unique `psk_identity` and `psk_key` (required for PSK). For **`[tls].mode = "mtls"`**, install the PEM set described under **Mutual TLS (`mtls`)** above at `/etc/nvidia-mnccd/tls/` on every node (private keys mode `0600`).

### Starting the service

After installation, load the unit and start MNCCD:

```sh
sudo systemctl daemon-reload
sudo systemctl start nvidia-mnccd
```

To start automatically on boot:

```sh
sudo systemctl enable nvidia-mnccd
```

Check status and logs:

```sh
systemctl status nvidia-mnccd
journalctl -u nvidia-mnccd.service -f
```

The bundled unit runs `ExecStart=/usr/bin/nvidia-mnccd` (no `--daemonize`); config is loaded from `/etc/nvidia-mnccd`. Transport mode comes from `[tls].mode` in that file—**TLS-PSK** when mode is `psk` or when `[tls].mode` is omitted (in which case `psk_identity` and `psk_key` must still be set).

### Logging

MNCCD logs through the [`tracing`](https://docs.rs/tracing) framework. Records carry one of five severities — `error`, `warn`, `info`, `debug`, `trace` — and are filtered against a configurable threshold.

**Log levels**

- Set the verbosity with `--log-level <error|warn|info|debug|trace>` (default `info`). This applies **only to MNCCD's own logs**; dependencies (tonic, hyper, rustls, etc.) stay at `info`, so `--log-level debug` gives you detailed MNCCD output without framework noise.
- The `RUST_LOG` environment variable, when set, **overrides** `--log-level` and controls **all** targets, with support for fine-grained per-module filters. For example:
  - `RUST_LOG=debug` — everything at `debug` and above, dependencies included.
  - `RUST_LOG=info,nvidia_mnccd=debug` — `debug` for MNCCD, `info` for dependencies (equivalent to `--log-level debug`).
- At `info` you get orchestration milestones (server start, NVLE setup/refresh phases, per-node results); `debug` adds per-RPC and per-node detail; `warn`/`error` surface retries, recovery events, and failures.

> **Note:** `debug` and `trace` are intended for troubleshooting. They increase log volume and may include request/response detail — avoid enabling them broadly in production.

**Where output goes**

| How MNCCD is run | Where output goes |
| --- | --- |
| systemd (`systemctl start nvidia-mnccd`) | **journald** — use `journalctl -u nvidia-mnccd.service` |
| `--daemonize` (manual) | `/tmp/nvidia-mnccd.out`, `/tmp/nvidia-mnccd.err` |
| Foreground (`cargo run`, direct binary) | Terminal stdout/stderr |

The output sink is selected automatically. Under systemd (`StandardOutput=journal`) MNCCD writes using journald's native protocol, so each tracing level maps to the matching journal priority (`error`→err, `warn`→warning, `info`→info, `debug`/`trace`→debug). This means severity-based filtering works, for example:

```sh
# Only warnings and errors from MNCCD
journalctl -u nvidia-mnccd.service -p warning
```

The log level must be chosen **before the service starts** — the unit is `Restart=no` and restarting MNCCD is not supported, so the level cannot be changed on a running instance. Set it while the service is stopped, then start it. To raise verbosity under systemd, add an environment override (systemd does not inherit your shell's `RUST_LOG`): run `sudo systemctl edit nvidia-mnccd.service` and add

```ini
[Service]
Environment=RUST_LOG=nvidia_mnccd=debug
```

then start the service and follow the logs:

```sh
sudo systemctl daemon-reload
sudo systemctl start nvidia-mnccd.service
journalctl -u nvidia-mnccd.service -f
```

In all other cases (foreground and `--daemonize`) MNCCD writes a human-readable text line per record (`timestamp LEVEL target: message`); ANSI colors are used only when stdout is an interactive terminal.

### API documentation

Generate Rust and gRPC reference docs into `docs/`:

```sh
./scripts/generate_docs.sh
```

Rust API docs: `docs/nvidia_mnccd/index.html`. gRPC messages: `docs/grpc_messages.html` (requires `protoc-gen-doc` on `PATH`).

## Contributions

This project is currently not accepting contributions.

