/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! NVIDIA Multi-Node Confidential Compute Daemon (MNCCD).
//!
//! MNCCD runs on every node in a multi-node GPU cluster. It exposes a gRPC API
//! for NVLink encryption (NVLE) orchestration: the leader node
//! confirms GPU fabric probe completion, validates ALID/CLID remap tables,
//! programs encryption keys, and locks remap tables before marking NVLE ready.
//!
//! # Configuration
//!
//! Cluster settings are loaded from `mnccd_config.toml` via [`init_config`] (see [`CONFIG`]). The
//! config directory is resolved from `MNCCD_DATA_DIR`, then `CARGO_MANIFEST_DIR`
//! (during `cargo run`), then the compile-time default (`/etc/nvidia-mnccd` unless
//! overridden at build time via `MNCCD_DEFAULT_DATA_DIR`).
//!
//! # TLS
//!
//! Set `[tls].mode` in `mnccd_config.toml` to `none`, `mtls`, or `psk` (default).
//! Call [`init_grpc_tls_mode`] once at startup (via [`resolve_grpc_tls_mode`]) before
//! spawning servers or clients. `--no-tls` forces plain mode on the CLI.
//! In `mtls` mode, the optional `[tls].server_name` and `[tls].allowed_client_organizations`
//! keys add peer certificate checks.
//!
//! # Service management
//!
//! Production deployments should run under systemd (foreground, logs in journald).
//! Pass `--daemonize` for manual background runs with stdout/stderr in `/tmp`.
//!
//! Only one MNCCD instance may run on a host; see [`acquire_instance_lock`].
//! Because that lock lives under `/run/nvidia-mnccd`, foreground runs require root;
//! non-root users must pass `--daemonize`, which locks a PID file under `/tmp`.
//!
//! # Logging
//!
//! Diagnostics are emitted through the [`tracing`] facade. Call [`init_logging`]
//! once at startup to install the global subscriber. The default verbosity is
//! chosen by the caller (e.g. the `--log-level` flag) and can be overridden with
//! the `RUST_LOG` environment variable for fine-grained per-module filtering.
//!
//! Under systemd (`StandardOutput=journal`) logs are written via journald's
//! native protocol, so tracing levels map to journal priorities; otherwise a
//! human-readable formatter writes to stdout. See [`init_logging`] for details.

use crate::libnvml_sys::{nvmlDeviceGpuRecoveryAction_t, safe_nvml};
use daemonize::Daemonize;
use futures::future::join_all;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::net::{IpAddr, SocketAddr};
use std::os::unix::io::AsRawFd;
use std::{
    error::Error,
    io::{ErrorKind, Write},
    path::PathBuf,
    sync::{LazyLock, Mutex, OnceLock},
};
use thiserror::Error;
use tonic::service::Interceptor;
use tonic::transport::server::TcpIncoming;
use tonic::transport::{
    Certificate, Channel, ClientTlsConfig, Endpoint, Identity, Server, ServerTlsConfig,
};
use tonic_tls::openssl::TlsIncoming;
use tracing::{debug, error, info, warn, Level};

mod crypto;
pub mod libnvml_sys;
mod mnccd_grpc;
mod node_allowlist;
mod peer_cert_policy;
mod psk_tls;
mod utils;

pub use mnccd_grpc::proto::MnccdGrpcPerGpuRemapTableInfo;
pub use utils::{is_leader, is_local};

use mnccd_grpc::proto::mnccd_grpc_client::MnccdGrpcClient;
use mnccd_grpc::proto::mnccd_grpc_server::MnccdGrpcServer;
use mnccd_grpc::{MnccdGlobalData, MnccdGrpcService};

/// syslog identifier used for journald entries (matches the systemd unit).
const JOURNALD_SYSLOG_IDENTIFIER: &str = "nvidia-mnccd";

/// Tracing target prefix for MNCCD's own crate (lib + binary share this name).
/// `CARGO_CRATE_NAME` is the crate name with dashes normalized to underscores,
/// which is exactly how `tracing` targets are spelled (unlike `CARGO_PKG_NAME`).
const CRATE_NAME: &str = env!("CARGO_CRATE_NAME");

/// Installs the process-wide [`tracing`] subscriber.
///
/// `default_level` is applied **only to MNCCD's own crate**; dependencies stay at
/// `info` so a `debug`/`trace` run doesn't drown MNCCD logs in framework noise.
/// This default is used when `RUST_LOG` is not set; when `RUST_LOG` is present it
/// takes precedence and controls all targets (e.g. `RUST_LOG=debug` enables
/// `debug` everywhere, dependencies included).
///
/// The output sink is chosen automatically:
///
/// * When started by systemd with `StandardOutput=journal` (detected via the
///   `JOURNAL_STREAM` environment variable), logs are written using journald's
///   native protocol. This maps each tracing level to the matching journal
///   `PRIORITY` (so `journalctl -p warning` and syslog severity filters work)
///   and lets journald supply the timestamp and identifier.
/// * Otherwise (interactive runs and `--daemonize`), a human-readable text
///   formatter writes to stdout.
///
/// `enable_ansi` only affects the text formatter; pass `false` whenever stdout is
/// redirected to a file so ANSI color escapes don't end up in the log file. It is
/// ignored for the journald sink.
///
/// Safe to call more than once (e.g. from tests); only the first call installs a
/// subscriber, subsequent calls are ignored.
pub fn init_logging(default_level: Level, enable_ansi: bool) {
    use tracing_subscriber::prelude::*;
    use tracing_subscriber::{fmt, EnvFilter, Registry};

    // Honor RUST_LOG when set (it controls every target); otherwise apply
    // `default_level` to MNCCD only and keep dependencies pinned at `info`.
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("info,{CRATE_NAME}={default_level}")));

    // systemd sets JOURNAL_STREAM when stdout/stderr are connected to the journal.
    // Prefer the native journald protocol there so levels become journal
    // priorities instead of plain text inside the message.
    let journald_layer = if std::env::var_os("JOURNAL_STREAM").is_some() {
        tracing_journald::layer()
            .map(|layer| layer.with_syslog_identifier(JOURNALD_SYSLOG_IDENTIFIER.to_string()))
            .ok()
    } else {
        None
    };

    // Fall back to the text formatter for interactive / `--daemonize` runs, or if
    // the journald socket couldn't be opened. `Option<Layer>` is a no-op when
    // `None`, so exactly one sink is active.
    let fmt_layer = if journald_layer.is_some() {
        None
    } else {
        Some(fmt::layer().with_ansi(enable_ansi).with_target(true))
    };

    let _ = Registry::default()
        .with(filter)
        .with(journald_layer)
        .with(fmt_layer)
        .try_init();
}

/// Generates cryptographically secure random bytes of the given length.
pub(crate) fn generate_random_bytes(len_in_bytes: usize) -> Result<Vec<u8>, getrandom::Error> {
    let mut buf = vec![0u8; len_in_bytes];
    getrandom::fill(&mut buf)?;
    Ok(buf)
}

#[derive(Debug, Deserialize)]
struct RawServer {
    server_port: u16,
}

#[derive(Debug, Deserialize)]
struct RawCluster {
    node_ips: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawRetryPolicy {
    /// Maximum number of retries for transient gRPC failures.
    max_retries: usize,
    /// Base delay for the first gRPC retry backoff, in milliseconds (doubles each attempt).
    initial_backoff_ms: u64,
    /// Upper bound on retry backoff delay, in milliseconds.
    max_backoff_ms: u64,
}

#[derive(Debug, Deserialize)]
struct RawTls {
    #[serde(default = "default_tls_mode")]
    mode: String,
    /// Required when `mode = "psk"`; empty/omitted is rejected at load for PSK mode.
    #[serde(default)]
    psk_identity: String,
    /// Required when `mode = "psk"`; empty/omitted is rejected at load for PSK mode.
    #[serde(default)]
    psk_key: String,
    /// Optional, `mtls` only: DNS name peer server certificates must present (default `mnccd`).
    #[serde(default)]
    server_name: String,
    /// Optional, `mtls` only: Subject Organization values accepted on peer client certificates.
    #[serde(default)]
    allowed_client_organizations: Vec<String>,
}

impl Default for RawTls {
    fn default() -> Self {
        Self {
            mode: default_tls_mode(),
            psk_identity: String::new(),
            psk_key: String::new(),
            server_name: String::new(),
            allowed_client_organizations: Vec::new(),
        }
    }
}

fn default_tls_mode() -> String {
    "psk".into()
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    server: RawServer,
    cluster: RawCluster,
    retry_policy: RawRetryPolicy,
    #[serde(default)]
    tls: RawTls,
}

/// gRPC transport security mode for MNCCD server and peer clients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrpcTlsMode {
    /// Plaintext HTTP/2 (`--no-tls` or `tls.mode = "none"`).
    Plain,
    /// Mutual TLS with operator-supplied PEM material under `tls/` (`tls.mode = "mtls"`).
    Mtls,
    /// OpenSSL TLS-PSK via tonic-tls (`tls.mode = "psk"`, default).
    Psk,
}

/// Parsed cluster configuration from `mnccd_config.toml`.
#[derive(Debug)]
pub struct Config {
    /// TCP port for the local gRPC server and outbound client connections.
    pub server_port: u16,
    /// IP address of the cluster leader (lowest address in `node_ips`).
    pub leader_ip: String,
    /// IP addresses of every node in the MNCCD cluster.
    pub node_ips: Vec<String>,
    /// Maximum retry attempts for transient gRPC failures.
    pub max_retries: usize,
    /// Initial exponential-backoff delay for gRPC retries, in milliseconds.
    pub initial_backoff_ms: u64,
    /// Upper bound on gRPC retry backoff delay, in milliseconds.
    pub max_backoff_ms: u64,
    /// gRPC TLS mode from `[tls].mode` in config (before `--no-tls` override).
    pub grpc_tls_mode: GrpcTlsMode,
    /// PSK credentials when `grpc_tls_mode` is [`GrpcTlsMode::Psk`]; `None` for `mtls` / `none`.
    pub(crate) psk_tls: Option<psk_tls::PskTlsConfig>,
    /// Optional mTLS peer certificate checks from `[tls].server_name` and
    /// `[tls].allowed_client_organizations`; the default turns both off.
    pub(crate) peer_cert_policy: peer_cert_policy::PeerCertPolicy,
}

/// Local GPU observed in RM recovery state for NVLE/link retrain handling.
#[derive(Debug, Clone)]
pub struct NvleRetrainGpu {
    /// Node that observed the recovery action.
    pub node_ip: String,
    /// UUID of the GPU in DRAIN_P2P.
    pub gpu_uuid: String,
    /// Local NVML device index on that node.
    pub gpu_index: u32,
    /// Raw NVML recovery action value.
    pub recovery_action: u32,
}

/// Compile-time default for the config directory. Packagers can override by building with
/// `MNCCD_DEFAULT_DATA_DIR=<path> cargo build`.
const DEFAULT_DATA_DIR: &str = match option_env!("MNCCD_DEFAULT_DATA_DIR") {
    Some(d) => d,
    None => "/etc/nvidia-mnccd",
};

/// Directory containing `mnccd_config.toml`. Resolution order:
/// 1. `MNCCD_DATA_DIR` env var (runtime override)
/// 2. `CARGO_MANIFEST_DIR` (set by Cargo during `cargo run`)
/// 3. `DEFAULT_DATA_DIR` (compile-time default)
pub(crate) fn mnccd_config_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("MNCCD_DATA_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(dir) = std::env::var("CARGO_MANIFEST_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(DEFAULT_DATA_DIR)
}

/// Directory containing TLS keys and certificates (`server.crt`, `client.key`, etc.).
pub(crate) fn mnccd_tls_dir() -> PathBuf {
    mnccd_config_dir().join("tls")
}

fn leader_ip_from_node_ips(node_ips: &[String]) -> Result<String, String> {
    if node_ips.is_empty() {
        return Err("node_ips must not be empty".to_string());
    }
    let mut min_ip: Option<IpAddr> = None;
    for s in node_ips {
        let ip = s
            .parse::<IpAddr>()
            .map_err(|e| format!("invalid IP in node_ips {s:?}: {e}"))?;
        min_ip = Some(match min_ip {
            Some(current) => current.min(ip),
            None => ip,
        });
    }
    Ok(min_ip.unwrap().to_string())
}

fn parse_grpc_tls_mode(mode: &str) -> Result<GrpcTlsMode, String> {
    match mode.trim().to_ascii_lowercase().as_str() {
        "none" => Ok(GrpcTlsMode::Plain),
        "mtls" => Ok(GrpcTlsMode::Mtls),
        "psk" => Ok(GrpcTlsMode::Psk),
        other => Err(format!(
            "invalid tls.mode '{other}': expected 'none', 'mtls', or 'psk'"
        )),
    }
}

fn load_config_fallible() -> Result<Config, String> {
    let dir_path = mnccd_config_dir();
    let file_content = std::fs::read_to_string(dir_path.join("mnccd_config.toml"))
        .map_err(|e| format!("failed to read mnccd_config.toml: {e}"))?;
    let raw: RawConfig = toml::from_str(&file_content)
        .map_err(|e| format!("failed to deserialize mnccd_config.toml: {e}"))?;
    let leader_ip = leader_ip_from_node_ips(&raw.cluster.node_ips)?;
    let grpc_tls_mode = parse_grpc_tls_mode(&raw.tls.mode)?;
    let psk_tls = if grpc_tls_mode == GrpcTlsMode::Psk {
        Some(psk_tls::PskTlsConfig::from_strings(
            &raw.tls.psk_identity,
            &raw.tls.psk_key,
        )?)
    } else {
        None
    };
    let peer_cert_policy = peer_cert_policy::PeerCertPolicy::from_config(
        &raw.tls.server_name,
        &raw.tls.allowed_client_organizations,
    )?;
    peer_cert_policy.ensure_supported(grpc_tls_mode)?;
    Ok(Config {
        server_port: raw.server.server_port,
        leader_ip,
        node_ips: raw.cluster.node_ips,
        max_retries: raw.retry_policy.max_retries,
        initial_backoff_ms: raw.retry_policy.initial_backoff_ms,
        max_backoff_ms: raw.retry_policy.max_backoff_ms,
        grpc_tls_mode,
        psk_tls,
        peer_cert_policy,
    })
}

static CONFIG_CELL: OnceLock<Config> = OnceLock::new();

/// Loads `mnccd_config.toml` once. Call from process startup before using [`CONFIG`].
///
/// Returns a descriptive error for operator-fixable problems (missing file, invalid
/// TOML, empty PSK credentials when `tls.mode = "psk"`, etc.) instead of panicking.
pub fn init_config() -> Result<(), String> {
    if CONFIG_CELL.get().is_some() {
        return Ok(());
    }
    let config = load_config_fallible()?;
    // A concurrent first call may have won the race; either way config is loaded.
    let _ = CONFIG_CELL.set(config);
    Ok(())
}

/// Handle that derefs to the process-wide [`Config`] after [`init_config`].
///
/// Accessing fields before `init_config` panics (programming error). Config *load*
/// failures are reported by `init_config` as `Err`.
#[derive(Debug)]
pub struct ConfigHandle;

impl std::ops::Deref for ConfigHandle {
    type Target = Config;

    fn deref(&self) -> &Config {
        CONFIG_CELL
            .get()
            .expect("internal error: init_config() must be called before accessing CONFIG")
    }
}

/// Process-wide cluster configuration. Initialize with [`init_config`] at startup.
pub static CONFIG: ConfigHandle = ConfigHandle;

static GRPC_TLS_MODE: OnceLock<GrpcTlsMode> = OnceLock::new();
static NVLE_KEY_REFRESH_LOCK: LazyLock<tokio::sync::Mutex<()>> =
    LazyLock::new(|| tokio::sync::Mutex::new(()));
static NVLE_SETUP_CONTEXT: LazyLock<Mutex<Option<NvleSetupContext>>> =
    LazyLock::new(|| Mutex::new(None));

const NVLE_KEY_SIZE_BYTES: usize = 32;
const NVLE_KEY_REFRESH_BUSY_MAX_ATTEMPTS: usize = 120;
const NVLE_KEY_REFRESH_BUSY_RETRY_DELAY_MS: u64 = 500;

#[derive(Clone)]
struct NvleSetupContext {
    gpu_infos: Vec<mnccd_grpc::proto::MnccdGrpcGpuInfo>,
    uuid_to_alid: HashMap<String, u32>,
}

fn cache_nvle_setup_context(context: NvleSetupContext) -> Result<(), String> {
    let mut cached_context = NVLE_SETUP_CONTEXT
        .lock()
        .map_err(|_| "NVLE setup context lock is poisoned".to_string())?;
    *cached_context = Some(context);
    Ok(())
}

fn get_cached_nvle_setup_context() -> Result<NvleSetupContext, String> {
    let cached_context = NVLE_SETUP_CONTEXT
        .lock()
        .map_err(|_| "NVLE setup context lock is poisoned".to_string())?;
    cached_context.clone().ok_or_else(|| {
        "NVLE setup context is unavailable; initial NVLE setup must complete before key refresh"
            .to_string()
    })
}

/// Resolves the active gRPC TLS mode from config and CLI (`--no-tls` forces plain).
pub fn resolve_grpc_tls_mode(no_tls_cli: bool) -> Result<GrpcTlsMode, String> {
    if no_tls_cli {
        return Ok(GrpcTlsMode::Plain);
    }
    Ok(CONFIG.grpc_tls_mode)
}

/// Warn when gRPC will run without TLS (`tls.mode = "none"` or `--no-tls`).
pub fn warn_if_plaintext_grpc(tls_mode: GrpcTlsMode) {
    if tls_mode == GrpcTlsMode::Plain {
        warn!(
            "gRPC TLS is disabled (tls.mode = \"none\" or --no-tls); \
             plaintext peer traffic can expose secrets—use only for bring-up, not production"
        );
    }
}

/// Sets gRPC transport security. Call once at startup before spawning servers or clients.
pub fn init_grpc_tls_mode(mode: GrpcTlsMode) {
    GRPC_TLS_MODE
        .set(mode)
        .expect("init_grpc_tls_mode must only be called once");
}

fn grpc_tls_mode() -> GrpcTlsMode {
    GRPC_TLS_MODE.get().copied().unwrap_or(GrpcTlsMode::Psk)
}

const RUNTIME_LOCK_DIR: &str = "/run/nvidia-mnccd";
const RUNTIME_LOCK_FILE: &str = "/run/nvidia-mnccd/nvidia-mnccd.lock";

/// Holds an exclusive flock on the instance lock file for the process lifetime.
pub struct InstanceLock {
    _file: File,
}

fn instance_lock_path() -> Result<PathBuf, std::io::Error> {
    std::fs::create_dir_all(RUNTIME_LOCK_DIR)?;
    Ok(PathBuf::from(RUNTIME_LOCK_FILE))
}

/// Acquires an exclusive non-blocking flock so only one MNCCD instance runs per host.
///
/// The lock file is always `/run/nvidia-mnccd/nvidia-mnccd.lock`. Under systemd,
/// `RuntimeDirectory=nvidia-mnccd` creates that directory before startup; for manual
/// runs, create it (or run as root) before starting the daemon. Non-root callers cannot
/// use this path and must run with `--daemonize` instead.
pub fn acquire_instance_lock() -> Result<InstanceLock, Box<dyn Error>> {
    let path = instance_lock_path()?;

    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&path)?;

    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if rc != 0 {
        let err = std::io::Error::last_os_error();
        if err.kind() == ErrorKind::WouldBlock {
            return Err(format!(
                "another nvidia-mnccd instance is already running (lock: {})",
                path.display()
            )
            .into());
        }
        return Err(err.into());
    }

    file.set_len(0)?;
    writeln!(file, "{}", std::process::id())?;

    Ok(InstanceLock { _file: file })
}

/// Detaches from the controlling terminal and redirects stdout/stderr to `/tmp`.
///
/// Writes the PID file to `/tmp/nvidia-mnccd.pid`. Used when `--daemonize` is passed;
/// the daemonize crate enforces a single instance via that PID file. Foreground and
/// systemd deployments should call [`acquire_instance_lock`] instead (see `main.rs`).
pub fn start_daemon() -> Result<(), Box<dyn Error>> {
    let stdout = File::create("/tmp/nvidia-mnccd.out").unwrap();
    let stderr = File::create("/tmp/nvidia-mnccd.err").unwrap();

    let daemonize = Daemonize::new()
        .pid_file("/tmp/nvidia-mnccd.pid")
        .working_directory("/tmp")
        .stdout(stdout)
        .stderr(stderr);

    daemonize
        .start()
        .map_err(|e| Box::new(e) as Box<dyn Error>)?;
    info!("Success, daemonized");
    Ok(())
}

/// Server-side checks for every gRPC request: the optional mTLS client Organization
/// check, then the `node_ips` allowlist.
fn server_interceptor(
    mut client_organization: Option<peer_cert_policy::ClientOrganizationInterceptor>,
    mut node_allowlist: node_allowlist::NodeAllowlistInterceptor,
) -> impl Interceptor + Clone + Send + Sync + 'static {
    move |request: tonic::Request<()>| -> Result<tonic::Request<()>, tonic::Status> {
        let request = match client_organization.as_mut() {
            Some(check) => check.call(request)?,
            None => request,
        };
        node_allowlist.call(request)
    }
}

/// Spawns the local gRPC server on this node's primary IP and [`CONFIG`] port.
///
/// Transport is selected by [`init_grpc_tls_mode`] (plain, mutual TLS, or TLS-PSK).
pub async fn spawn_and_start_server(
) -> Result<tokio::task::JoinHandle<Result<(), Box<dyn Error + Send + Sync>>>, Box<dyn Error>> {
    let local_ip = utils::local_node_ip()?;
    let server_addr = SocketAddr::new(local_ip, CONFIG.server_port);
    let tls_mode = grpc_tls_mode();

    let mtls_config = if tls_mode == GrpcTlsMode::Mtls {
        let data_dir = mnccd_tls_dir();
        let cert = tokio::fs::read_to_string(data_dir.join("server.crt"))
            .await
            .map_err(|e| format!("unable to read server certificate: {e}"))?;
        let key = tokio::fs::read_to_string(data_dir.join("server.key"))
            .await
            .map_err(|e| format!("unable to read server private key: {e}"))?;
        let client_ca = tokio::fs::read_to_string(data_dir.join("ca-root.crt"))
            .await
            .map_err(|e| format!("unable to read CA root certificate: {e}"))?;

        Some(
            ServerTlsConfig::new()
                .identity(Identity::from_pem(cert, key))
                .client_ca_root(Certificate::from_pem(client_ca)),
        )
    } else {
        None
    };

    let psk_acceptor = if tls_mode == GrpcTlsMode::Psk {
        let psk_cfg = CONFIG.psk_tls.as_ref().ok_or_else(|| {
            std::io::Error::new(
                ErrorKind::InvalidInput,
                "grpc_tls_mode is Psk but [tls].psk_identity/psk_key are unavailable",
            )
        })?;
        Some(
            psk_tls::server_acceptor(psk_cfg)
                .map_err(|e| format!("invalid PSK TLS server config: {e}"))?,
        )
    } else {
        None
    };

    let interceptor = node_allowlist::NodeAllowlistInterceptor::from_node_ips(&CONFIG.node_ips)
        .map_err(|e| format!("failed to build node IP allowlist: {e}"))?;
    let interceptor = server_interceptor(
        CONFIG.peer_cert_policy.client_organization_interceptor(tls_mode),
        interceptor,
    );

    let t1 = tokio::spawn(async move {
        let global_data = MnccdGlobalData::new();
        let service = MnccdGrpcServer::with_interceptor(
            MnccdGrpcService::new(global_data.clone()),
            interceptor,
        );

        let serve_result = match tls_mode {
            GrpcTlsMode::Mtls => {
                let tls =
                    mtls_config.expect("mtls_config must be present when grpc_tls_mode is Mtls");
                info!("Starting server on {server_addr} with mutual TLS...");
                let mut builder = Server::builder()
                    .tls_config(tls)
                    .map_err(|e| format!("invalid TLS config: {e}"))?;
                builder
                    .add_service(service)
                    .serve(server_addr)
                    .await
                    .map_err(|e| format!("gRPC server failed: {e}"))
            }
            GrpcTlsMode::Psk => {
                let acceptor =
                    psk_acceptor.expect("psk_acceptor must be present when grpc_tls_mode is Psk");
                info!("Starting server on {server_addr} with TLS-PSK...");
                let incoming = TlsIncoming::new(
                    TcpIncoming::bind(server_addr)
                        .map_err(|e| format!("failed to bind {server_addr}: {e}"))?,
                    acceptor,
                );
                Server::builder()
                    .add_service(service)
                    .serve_with_incoming(incoming)
                    .await
                    .map_err(|e| format!("gRPC server failed: {e}"))
            }
            GrpcTlsMode::Plain => {
                info!("Starting server on {server_addr} without TLS...");
                Server::builder()
                    .add_service(service)
                    .serve(server_addr)
                    .await
                    .map_err(|e| format!("gRPC server failed: {e}"))
            }
        };

        serve_result.map_err(|e| -> Box<dyn Error + Send + Sync> { e.into() })
    });

    Ok(t1)
}

/// Failure while bringing up a gRPC client to a cluster peer.
#[derive(Debug, Error)]
pub enum ClientStartError {
    #[error("failed to create channel to peer {node_addr}")]
    Connect {
        node_addr: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("mnccd_grpc_echo_message to peer {node_addr} failed after retries")]
    Echo {
        node_addr: String,
        #[source]
        source: tonic::Status,
    },
}

impl ClientStartError {
    /// Cluster peer address (IP or hostname) from config.
    pub fn node_addr(&self) -> &str {
        match self {
            Self::Connect { node_addr, .. } | Self::Echo { node_addr, .. } => node_addr,
        }
    }
}

/// Spawns a task that connects to `client_addr`, runs the echo RPC, and retries on failure.
pub fn spawn_and_start_client(
    client_addr: &String,
) -> tokio::task::JoinHandle<Result<(), ClientStartError>> {
    let tls_mode = grpc_tls_mode();
    let client_addr = client_addr.clone();

    tokio::spawn(async move {
        let channel = build_client_channel(&client_addr, tls_mode)
            .await
            .map_err(|source| ClientStartError::Connect {
                node_addr: client_addr.clone(),
                source,
            })?;

        let req_data = mnccd_grpc::proto::MnccdGrpcEchoRequest {
            name: "MNCCD client".into(),
        };

        let channel_for_retry = channel.clone();
        let response = mnccd_grpc::call_with_retry(
            move || {
                let channel = channel_for_retry.clone();
                let req = tonic::Request::new(req_data.clone());
                Box::pin(async move {
                    let mut client = MnccdGrpcClient::new(channel);
                    client.mnccd_grpc_echo_message(req).await
                })
            },
            "mnccd_grpc_echo_message",
        )
        .await
        .map_err(|source| ClientStartError::Echo {
            node_addr: client_addr.clone(),
            source,
        })?;
        debug!("RESPONSE={response:?}");
        Ok(())
    })
}

/// Formats a host for a gRPC URI authority. IPv6 literals are bracketed per RFC 3986.
fn format_host_for_uri_authority(host: &str) -> String {
    if let Ok(ip) = host.parse::<IpAddr>() {
        return match ip {
            IpAddr::V4(_) => ip.to_string(),
            IpAddr::V6(addr) => format!("[{addr}]"),
        };
    }
    host.to_string()
}

/// Builds `scheme://host:port` for tonic `Endpoint::from_shared`, with correct IPv6 syntax.
fn build_client_endpoint_url(scheme: &str, host: &str, port: u16) -> String {
    format!("{scheme}://{}:{port}", format_host_for_uri_authority(host))
}

/// Creates a channel handle for connecting to a cluster peer.
pub async fn create_channel_handle_for_client(
    client_addr: &String,
) -> Result<Channel, Box<dyn std::error::Error + Send + Sync>> {
    build_client_channel(client_addr, grpc_tls_mode()).await
}

async fn build_client_channel(
    client_addr: &str,
    tls_mode: GrpcTlsMode,
) -> Result<Channel, Box<dyn std::error::Error + Send + Sync>> {
    match tls_mode {
        GrpcTlsMode::Plain => {
            let url = build_client_endpoint_url("http", client_addr, CONFIG.server_port);
            debug!("Creating channel for {url} using non-TLS connection...");
            Ok(Channel::from_shared(url)
                .map_err(|e| format!("invalid client URL: {e}"))?
                .connect_lazy())
        }
        GrpcTlsMode::Mtls => {
            let url = build_client_endpoint_url("https", client_addr, CONFIG.server_port);
            let data_dir = mnccd_tls_dir();
            let server_root_ca_cert = tokio::fs::read_to_string(data_dir.join("ca-root.crt"))
                .await
                .map_err(|e| format!("unable to read CA root certificate: {e}"))?;
            let server_root_ca_cert = Certificate::from_pem(server_root_ca_cert);

            let client_cert = tokio::fs::read_to_string(data_dir.join("client.crt"))
                .await
                .map_err(|e| format!("unable to read client certificate: {e}"))?;
            let client_key = tokio::fs::read_to_string(data_dir.join("client.key"))
                .await
                .map_err(|e| format!("unable to read client private key: {e}"))?;
            let client_identity = Identity::from_pem(client_cert, client_key);

            let tls = ClientTlsConfig::new()
                .domain_name(CONFIG.peer_cert_policy.server_name())
                .ca_certificate(server_root_ca_cert)
                .identity(client_identity);

            debug!("Creating channel for {url} using mutual TLS...");
            Ok(Channel::from_shared(url)
                .map_err(|e| format!("invalid client URL: {e}"))?
                .tls_config(tls)
                .map_err(|e| format!("invalid TLS config: {e}"))?
                .connect_lazy())
        }
        GrpcTlsMode::Psk => {
            let psk_cfg = CONFIG.psk_tls.as_ref().ok_or_else(|| {
                std::io::Error::new(
                    ErrorKind::InvalidInput,
                    "grpc_tls_mode is Psk but [tls].psk_identity/psk_key are unavailable",
                )
            })?;
            let url = build_client_endpoint_url("http", client_addr, CONFIG.server_port);
            debug!("Creating channel for {url} using TLS-PSK...");
            let endpoint =
                Endpoint::from_shared(url).map_err(|e| format!("invalid client URL: {e}"))?;
            let ssl = psk_tls::client_connector(psk_cfg)
                .map_err(|e| format!("invalid PSK TLS client config: {e}"))?;
            let transport = tonic_tls::TcpTransport::from_endpoint(&endpoint);
            let peer_name = format_host_for_uri_authority(client_addr);
            Ok(
                endpoint.connect_with_connector_lazy(tonic_tls::openssl::TlsConnector::new(
                    transport, ssl, peer_name,
                )),
            )
        }
    }
}

/// Polls local RM/NVML state and returns GPUs currently asking for P2P drain.
pub fn collect_local_drain_p2p_recovery_gpus() -> Result<Vec<NvleRetrainGpu>, String> {
    safe_nvml::init().map_err(|e| format!("Failed to initialize NVML: {e}"))?;

    let result = (|| {
        let num_gpus =
            safe_nvml::device_get_count().map_err(|e| format!("Failed to get GPU count: {e}"))?;
        let node_ip = utils::local_node_ip()?.to_string();
        let drain_p2p = nvmlDeviceGpuRecoveryAction_t::NVML_GPU_RECOVERY_ACTION_DRAIN_P2P as u32;

        let mut gpus = Vec::new();
        for gpu_index in 0..num_gpus {
            let device = safe_nvml::Device::new_from_index(gpu_index)
                .map_err(|e| format!("Failed to get device handle for GPU {gpu_index}: {e}"))?;
            let recovery_action = safe_nvml::device_get_gpu_recovery_action(&device)
                .map_err(|e| format!("Failed to get recovery action for GPU {gpu_index}: {e}"))?;

            if recovery_action == drain_p2p {
                let gpu_uuid = device
                    .get_uuid()
                    .map_err(|e| format!("Failed to get UUID for GPU {gpu_index}: {e}"))?;
                gpus.push(NvleRetrainGpu {
                    node_ip: node_ip.clone(),
                    gpu_uuid,
                    gpu_index,
                    recovery_action,
                });
            }
        }

        Ok(gpus)
    })();

    let _ = safe_nvml::shutdown();
    result
}

async fn report_nvle_retrain_event_to_leader(gpus: Vec<NvleRetrainGpu>) -> Result<(), String> {
    let channel = create_channel_handle_for_client(&CONFIG.leader_ip)
        .await
        .map_err(|e| {
            format!(
                "Failed to create channel to leader {}: {e}",
                CONFIG.leader_ip
            )
        })?;

    let req_data = mnccd_grpc::proto::MnccdGrpcReportRetrainEventRequest {
        gpus: gpus
            .into_iter()
            .map(|gpu| mnccd_grpc::proto::MnccdGrpcRetrainGpu {
                node_ip: gpu.node_ip,
                gpu_uuid: gpu.gpu_uuid,
                gpu_index: gpu.gpu_index,
                recovery_action: gpu.recovery_action,
            })
            .collect(),
    };

    let channel_for_retry = channel.clone();
    let response = mnccd_grpc::call_with_retry(
        move || {
            let channel = channel_for_retry.clone();
            let req = tonic::Request::new(req_data.clone());
            Box::pin(async move {
                let mut client = MnccdGrpcClient::new(channel);
                client.mnccd_grpc_report_retrain_event(req).await
            })
        },
        "mnccd_grpc_report_retrain_event",
    )
    .await
    .map_err(|e| format!("Failed to report retrain event to leader: {e}"))?;

    let inner = response.into_inner();
    if !inner.success {
        return Err(format!(
            "Leader failed to handle retrain event: {}",
            inner.error_message
        ));
    }

    Ok(())
}

/// Handles a DRAIN_P2P recovery signal observed by the local MNCCD instance.
pub async fn handle_nvle_retrain_event(gpus: Vec<NvleRetrainGpu>) -> Result<(), String> {
    if gpus.is_empty() {
        return Ok(());
    }

    if utils::is_leader()? {
        let trigger_gpu_uuids = gpus.iter().map(|gpu| gpu.gpu_uuid.clone()).collect();
        info!(
            "Leader handling NVLE retrain event for {} local GPU(s)",
            gpus.len()
        );
        refresh_nvle_keys_for_drain_p2p_event(trigger_gpu_uuids).await
    } else {
        info!(
            "Reporting NVLE retrain event for {} local GPU(s) to leader {}",
            gpus.len(),
            CONFIG.leader_ip
        );
        report_nvle_retrain_event_to_leader(gpus).await
    }
}

/// Starts a background polling WAR for RM DRAIN_P2P recovery-action events.
pub fn spawn_nvle_retrain_monitor() -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut active_drain_uuids = HashSet::new();

        loop {
            let poll_result = tokio::task::spawn_blocking(collect_local_drain_p2p_recovery_gpus)
                .await
                .map_err(|e| format!("NVLE retrain monitor task join failed: {e}"))
                .and_then(|result| result);

            match poll_result {
                Ok(gpus) => {
                    let current_drain_uuids: HashSet<String> =
                        gpus.iter().map(|gpu| gpu.gpu_uuid.clone()).collect();
                    let new_gpus: Vec<NvleRetrainGpu> = gpus
                        .into_iter()
                        .filter(|gpu| !active_drain_uuids.contains(&gpu.gpu_uuid))
                        .collect();

                    if !new_gpus.is_empty() {
                        warn!(
                            "Detected RM DRAIN_P2P recovery action for GPU(s): {:?}",
                            new_gpus
                                .iter()
                                .map(|gpu| gpu.gpu_uuid.as_str())
                                .collect::<Vec<_>>()
                        );

                        match handle_nvle_retrain_event(new_gpus.clone()).await {
                            Ok(()) => {
                                active_drain_uuids
                                    .extend(new_gpus.into_iter().map(|gpu| gpu.gpu_uuid));
                            }
                            Err(e) => {
                                error!("Failed to handle NVLE retrain event: {e}");
                            }
                        }
                    }

                    active_drain_uuids.retain(|uuid| current_drain_uuids.contains(uuid));
                }
                Err(e) => warn!("Failed to poll RM recovery action: {e}"),
            }

            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        }
    })
}

#[derive(Clone)]
struct NvleKeySetupJob {
    channel: Channel,
    device_uuid: String,
    /// Cluster IP of the node that owns `device_uuid` (the RPC target).
    node_ip: String,
    local_gpu_alid: u32,
    remote_gpu_alid: u32,
    local_gpu_nvle_private_info: Vec<u8>,
    remote_gpu_nvle_private_info: Vec<u8>,
    nvle_key: Vec<u8>,
}

#[derive(Clone)]
struct NvleKeySetupPair {
    local: NvleKeySetupJob,
    remote: NvleKeySetupJob,
    /// True when both jobs target the same node. Such pairs must be programmed
    /// serially, since concurrent RPCs to one node drive concurrent NVML calls
    /// on that host; cross-node pairs can be programmed in parallel.
    same_node: bool,
}

fn nvle_key_setup_response_is_busy(
    response: &mnccd_grpc::proto::MnccdGrpcSetupNvleEncryptionKeyResponse,
) -> bool {
    response.retryable
}

fn key_setup_job_error(job: &NvleKeySetupJob, message: impl std::fmt::Display) -> String {
    format!(
        "Failed to setup encryption key for local ALID {} remote ALID {} on {} (node {}): {}",
        job.local_gpu_alid, job.remote_gpu_alid, job.device_uuid, job.node_ip, message
    )
}

async fn setup_nvle_encryption_key_rpc_once(
    job: NvleKeySetupJob,
) -> Result<mnccd_grpc::proto::MnccdGrpcSetupNvleEncryptionKeyResponse, tonic::Status> {
    let response = setup_nvle_encryption_key_rpc(
        job.channel,
        job.device_uuid,
        job.local_gpu_alid,
        job.remote_gpu_alid,
        job.local_gpu_nvle_private_info,
        job.remote_gpu_nvle_private_info,
        job.nvle_key,
    )
    .await?;
    Ok(response.into_inner())
}

async fn program_nvle_key_refresh_jobs(
    mut pending_pairs: Vec<NvleKeySetupPair>,
) -> Result<(), String> {
    let mut busy_attempts = 0usize;

    while !pending_pairs.is_empty() {
        let mut still_busy = Vec::new();

        for pair in pending_pairs.into_iter() {
            let (local_result, remote_result) = if pair.same_node {
                // Same-node pair: serialize to avoid concurrent NVML calls on the host.
                let local_result = setup_nvle_encryption_key_rpc_once(pair.local.clone()).await;
                let remote_result = setup_nvle_encryption_key_rpc_once(pair.remote.clone()).await;
                (local_result, remote_result)
            } else {
                // Cross-node pair: the RPCs land on different hosts, so run them concurrently.
                tokio::join!(
                    setup_nvle_encryption_key_rpc_once(pair.local.clone()),
                    setup_nvle_encryption_key_rpc_once(pair.remote.clone()),
                )
            };

            let local_busy = match local_result {
                Ok(inner) if inner.success => false,
                Ok(inner) if nvle_key_setup_response_is_busy(&inner) => true,
                Ok(inner) => return Err(key_setup_job_error(&pair.local, inner.error_message)),
                Err(e) => return Err(key_setup_job_error(&pair.local, e)),
            };

            let remote_busy = match remote_result {
                Ok(inner) if inner.success => false,
                Ok(inner) if nvle_key_setup_response_is_busy(&inner) => true,
                Ok(inner) => return Err(key_setup_job_error(&pair.remote, inner.error_message)),
                Err(e) => return Err(key_setup_job_error(&pair.remote, e)),
            };

            if local_busy || remote_busy {
                still_busy.push(pair);
            }
        }

        if still_busy.is_empty() {
            return Ok(());
        }

        busy_attempts += 1;
        if busy_attempts >= NVLE_KEY_REFRESH_BUSY_MAX_ATTEMPTS {
            let uuids = still_busy
                .iter()
                .flat_map(|pair| {
                    [
                        format!("{} (node {})", pair.local.device_uuid, pair.local.node_ip),
                        format!("{} (node {})", pair.remote.device_uuid, pair.remote.node_ip),
                    ]
                })
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "Timed out waiting for NVLE key refresh to leave NVML_ERROR_IN_USE after {} attempts: {}",
                NVLE_KEY_REFRESH_BUSY_MAX_ATTEMPTS, uuids
            ));
        }

        warn!(
            "{} NVLE key refresh GPU pair(s) still busy; retrying attempt {} of {}",
            still_busy.len(),
            busy_attempts + 1,
            NVLE_KEY_REFRESH_BUSY_MAX_ATTEMPTS
        );
        tokio::time::sleep(std::time::Duration::from_millis(
            NVLE_KEY_REFRESH_BUSY_RETRY_DELAY_MS,
        ))
        .await;
        pending_pairs = still_busy;
    }

    Ok(())
}

async fn create_channel_map_for_configured_nodes() -> Result<HashMap<String, Channel>, String> {
    // Channel setup is pure gRPC connection establishment (no NVML) and is
    // independent per node, so connect to all configured nodes concurrently.
    let connect_tasks = CONFIG.node_ips.iter().map(|node_ip| {
        let node_ip = node_ip.clone();
        async move {
            debug!("Creating channel to node {}...", node_ip);
            let channel = create_channel_handle_for_client(&node_ip)
                .await
                .map_err(|e| format!("Failed to create channel to {}: {}", node_ip, e))?;
            Ok::<(String, Channel), String>((node_ip, channel))
        }
    });

    let mut channel_map = HashMap::new();
    for result in join_all(connect_tasks).await {
        let (node_ip, channel) = result?;
        channel_map.insert(node_ip, channel);
    }
    Ok(channel_map)
}

async fn collect_gpu_infos_from_all_nodes(
    channel_map: &HashMap<String, Channel>,
    log_prefix: &str,
) -> Result<Vec<mnccd_grpc::proto::MnccdGrpcGpuInfo>, String> {
    use mnccd_grpc::proto;

    info!("{log_prefix}: Collecting GPU information from all nodes...");
    let mut collect_tasks = Vec::new();
    let mut collect_errors: Vec<String> = Vec::new();
    for node_ip in &CONFIG.node_ips {
        let node_ip = node_ip.clone();
        let channel = channel_map
            .get(&node_ip)
            .ok_or_else(|| format!("Channel not found for node {}", node_ip))?
            .clone();

        collect_tasks.push(async move {
            debug!("Collecting GPU info from node {}...", node_ip);
            let channel_clone = channel.clone();
            let response = mnccd_grpc::call_with_retry(
                move || {
                    let channel = channel_clone.clone();
                    let req = tonic::Request::new(proto::MnccdGrpcCollectGpusInfoRequest {});
                    Box::pin(async move {
                        let mut client = MnccdGrpcClient::new(channel);
                        client.mnccd_grpc_collect_gpus_info(req).await
                    })
                },
                "mnccd_grpc_collect_gpus_info",
            )
            .await
            .map_err(|e| format!("Failed to collect GPU info from {}: {}", node_ip, e))?;

            let response_inner = response.into_inner();
            info!(
                "Found {} GPUs with active links on node {}",
                response_inner.gpu_infos.len(),
                node_ip
            );

            let mut gpu_infos = response_inner.gpu_infos;
            for gpu_info in &mut gpu_infos {
                gpu_info.node_ip = node_ip.clone();
            }

            Ok(gpu_infos)
        });
    }

    let mut all_gpu_infos = Vec::new();
    for result in join_all(collect_tasks).await {
        match result {
            Ok(gpu_infos) => all_gpu_infos.extend(gpu_infos),
            Err(e) => collect_errors.push(e),
        }
    }

    if !collect_errors.is_empty() {
        return Err(collect_errors.join("; "));
    }
    if all_gpu_infos.is_empty() {
        return Err("No GPUs with active links found on any node".to_string());
    }

    info!(
        "{log_prefix}: Total GPUs collected from all nodes: {}",
        all_gpu_infos.len()
    );
    Ok(all_gpu_infos)
}

async fn collect_remap_table_info_from_all_nodes(
    channel_map: &HashMap<String, Channel>,
    log_prefix: &str,
) -> Result<Vec<mnccd_grpc::proto::MnccdGrpcPerGpuRemapTableInfo>, String> {
    use mnccd_grpc::proto;

    info!("{log_prefix}: Querying remap tables on all GPUs on every node...");
    let mut query_tasks = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    for node_ip in &CONFIG.node_ips {
        let node_ip = node_ip.clone();
        let channel = channel_map
            .get(&node_ip)
            .ok_or_else(|| format!("Channel not found for node {}", node_ip))?
            .clone();

        query_tasks.push(async move {
            debug!("Querying remap table on node {}...", node_ip);
            let channel_clone = channel.clone();
            let response = mnccd_grpc::call_with_retry(
                move || {
                    let channel = channel_clone.clone();
                    let req = tonic::Request::new(proto::MnccdGrpcQueryRemapTableRequest {});
                    Box::pin(async move {
                        let mut client = MnccdGrpcClient::new(channel);
                        client.mnccd_grpc_query_remap_table(req).await
                    })
                },
                "mnccd_grpc_query_remap_table",
            )
            .await
            .map_err(|e| format!("Failed to query remap table on {}: {}", node_ip, e))?;

            let inner = response.into_inner();
            if !inner.success {
                return Err(format!(
                    "Failed to query remap table on {}: {}",
                    node_ip, inner.error_message
                ));
            }

            info!(
                "Remap table queried for {} GPUs on node {}",
                inner.queried_gpu_uuids.len(),
                node_ip
            );
            Ok(inner.per_gpu_remap_table)
        });
    }

    let mut all_per_gpu_remap_table = Vec::new();
    for result in join_all(query_tasks).await {
        match result {
            Ok(per_gpu) => all_per_gpu_remap_table.extend(per_gpu),
            Err(e) => errors.push(e),
        }
    }

    if !errors.is_empty() {
        return Err(format!(
            "Remap table query completed with errors: {}",
            errors.join("; ")
        ));
    }

    info!(
        "{log_prefix}: Remap table queries complete on all nodes ({} GPU entries)",
        all_per_gpu_remap_table.len()
    );
    Ok(all_per_gpu_remap_table)
}

fn build_uuid_to_alid_map(
    per_gpu_remap_table: &[mnccd_grpc::proto::MnccdGrpcPerGpuRemapTableInfo],
) -> Result<HashMap<String, u32>, String> {
    let mut uuid_to_alid = HashMap::new();
    let mut alid_to_uuid = HashMap::new();
    for entry in per_gpu_remap_table {
        if let Some(existing_uuid) = alid_to_uuid.get(&entry.alid) {
            if existing_uuid != &entry.gpu_uuid {
                return Err(format!(
                    "Duplicate ALID {} resolved for GPUs {} and {}; ALIDs must be unique across the fabric",
                    entry.alid, existing_uuid, entry.gpu_uuid
                ));
            }
        }
        alid_to_uuid.insert(entry.alid, entry.gpu_uuid.clone());
        uuid_to_alid.insert(entry.gpu_uuid.clone(), entry.alid);
    }
    Ok(uuid_to_alid)
}

fn validate_remap_table_coverage(
    gpu_infos: &[mnccd_grpc::proto::MnccdGrpcGpuInfo],
    per_gpu_remap_table: &[mnccd_grpc::proto::MnccdGrpcPerGpuRemapTableInfo],
) -> Result<(), String> {
    let expected_gpu_uuids: HashSet<String> =
        gpu_infos.iter().map(|gpu| gpu.uuid.clone()).collect();
    let fabric_gpu_count = expected_gpu_uuids.len();
    let mut seen_gpu_uuids = HashSet::new();

    for entry in per_gpu_remap_table {
        if !expected_gpu_uuids.contains(&entry.gpu_uuid) {
            return Err(format!(
                "Remap table query returned unexpected GPU UUID {}",
                entry.gpu_uuid
            ));
        }
        if !seen_gpu_uuids.insert(entry.gpu_uuid.clone()) {
            return Err(format!(
                "Remap table query returned duplicate entry for GPU UUID {}",
                entry.gpu_uuid
            ));
        }

        let remap_tab_size = entry.remap_tab_size as usize;
        if remap_tab_size < fabric_gpu_count {
            return Err(format!(
                "GPU {} remap table has {} entries, but fabric has {} GPUs",
                entry.gpu_uuid, remap_tab_size, fabric_gpu_count
            ));
        }
        if entry.fla_remap_table_addr.len() != remap_tab_size
            || entry.gpa_remap_table_addr.len() != remap_tab_size
        {
            return Err(format!(
                "GPU {} remap table size {} does not match FLA/GPA vector lengths {}/{}",
                entry.gpu_uuid,
                remap_tab_size,
                entry.fla_remap_table_addr.len(),
                entry.gpa_remap_table_addr.len()
            ));
        }
    }

    let missing_gpu_uuids: Vec<String> = expected_gpu_uuids
        .difference(&seen_gpu_uuids)
        .cloned()
        .collect();
    if !missing_gpu_uuids.is_empty() {
        return Err(format!(
            "Missing remap table response for {} GPU(s): {}",
            missing_gpu_uuids.len(),
            missing_gpu_uuids.join(", ")
        ));
    }

    Ok(())
}

fn build_nvle_key_setup_jobs(
    gpu_infos: &[mnccd_grpc::proto::MnccdGrpcGpuInfo],
    uuid_to_alid: &HashMap<String, u32>,
    channel_map: &HashMap<String, Channel>,
) -> Result<Vec<NvleKeySetupPair>, String> {
    let mut jobs = Vec::new();

    for i in 0..gpu_infos.len() {
        for j in (i + 1)..gpu_infos.len() {
            let local_gpu = &gpu_infos[i];
            let remote_gpu = &gpu_infos[j];
            let key_secret = generate_random_bytes(NVLE_KEY_SIZE_BYTES).map_err(|e| {
                format!(
                    "Failed to generate encryption key secret for GPU pair {} <-> {}: {}",
                    local_gpu.uuid, remote_gpu.uuid, e
                )
            })?;

            let local_alid = *uuid_to_alid
                .get(&local_gpu.uuid)
                .ok_or_else(|| format!("ALID not found for GPU {}", local_gpu.uuid))?;
            let remote_alid = *uuid_to_alid
                .get(&remote_gpu.uuid)
                .ok_or_else(|| format!("ALID not found for GPU {}", remote_gpu.uuid))?;
            let local_channel = channel_map
                .get(&local_gpu.node_ip)
                .ok_or_else(|| format!("Channel not found for node {}", local_gpu.node_ip))?
                .clone();
            let remote_channel = channel_map
                .get(&remote_gpu.node_ip)
                .ok_or_else(|| format!("Channel not found for node {}", remote_gpu.node_ip))?
                .clone();

            let local_job = NvleKeySetupJob {
                channel: local_channel,
                device_uuid: local_gpu.uuid.clone(),
                node_ip: local_gpu.node_ip.clone(),
                local_gpu_alid: local_alid,
                remote_gpu_alid: remote_alid,
                local_gpu_nvle_private_info: local_gpu.nvle_private_info.clone(),
                remote_gpu_nvle_private_info: remote_gpu.nvle_private_info.clone(),
                nvle_key: key_secret.clone(),
            };
            let remote_job = NvleKeySetupJob {
                channel: remote_channel,
                device_uuid: remote_gpu.uuid.clone(),
                node_ip: remote_gpu.node_ip.clone(),
                local_gpu_alid: remote_alid,
                remote_gpu_alid: local_alid,
                local_gpu_nvle_private_info: remote_gpu.nvle_private_info.clone(),
                remote_gpu_nvle_private_info: local_gpu.nvle_private_info.clone(),
                nvle_key: key_secret,
            };
            jobs.push(NvleKeySetupPair {
                local: local_job,
                remote: remote_job,
                same_node: local_gpu.node_ip == remote_gpu.node_ip,
            });
        }
    }

    Ok(jobs)
}

async fn collect_drain_p2p_recovery_gpu_uuids_from_all_nodes(
    channel_map: &HashMap<String, Channel>,
) -> Result<HashSet<String>, String> {
    use mnccd_grpc::proto;

    info!("Collecting RM DRAIN_P2P recovery state from all nodes...");
    let mut collect_tasks = Vec::new();
    let mut collect_errors: Vec<String> = Vec::new();
    for node_ip in &CONFIG.node_ips {
        let node_ip = node_ip.clone();
        let channel = channel_map
            .get(&node_ip)
            .ok_or_else(|| format!("Channel not found for node {}", node_ip))?
            .clone();

        collect_tasks.push(async move {
            let channel_clone = channel.clone();
            let response = mnccd_grpc::call_with_retry(
                move || {
                    let channel = channel_clone.clone();
                    let req = tonic::Request::new(proto::MnccdGrpcCollectDrainP2pGpusRequest {});
                    Box::pin(async move {
                        let mut client = MnccdGrpcClient::new(channel);
                        client.mnccd_grpc_collect_drain_p2p_gpus(req).await
                    })
                },
                "mnccd_grpc_collect_drain_p2p_gpus",
            )
            .await
            .map_err(|e| format!("Failed to collect DRAIN_P2P GPUs from {}: {}", node_ip, e))?;

            let mut inner = response.into_inner();
            if !inner.success {
                return Err(format!(
                    "Failed to collect DRAIN_P2P GPUs from {}: {}",
                    node_ip, inner.error_message
                ));
            }

            for gpu in &mut inner.gpus {
                gpu.node_ip = node_ip.clone();
            }

            info!(
                "Node {} reports {} GPU(s) in DRAIN_P2P",
                node_ip,
                inner.gpus.len()
            );

            Ok::<Vec<proto::MnccdGrpcRetrainGpu>, String>(inner.gpus)
        });
    }

    let mut drained_gpu_uuids = HashSet::new();
    for result in join_all(collect_tasks).await {
        match result {
            Ok(gpus) => {
                for gpu in gpus {
                    drained_gpu_uuids.insert(gpu.gpu_uuid);
                }
            }
            Err(e) => collect_errors.push(e),
        }
    }

    if !collect_errors.is_empty() {
        return Err(collect_errors.join("; "));
    }

    info!(
        "Observed {} GPU(s) in RM DRAIN_P2P recovery state",
        drained_gpu_uuids.len()
    );
    Ok(drained_gpu_uuids)
}

fn sorted_set_difference(left: &HashSet<String>, right: &HashSet<String>) -> Vec<String> {
    let mut difference: Vec<String> = left.difference(right).cloned().collect();
    difference.sort_unstable();
    difference
}

async fn set_nvle_ready_on_node(
    channel: Channel,
    node_ip: String,
    expected_uuids: HashSet<String>,
    ready: bool,
    log_prefix: String,
) -> Result<String, String> {
    use mnccd_grpc::proto;

    debug!(
        "{log_prefix}: Setting NVLE ready state to {} on node {}...",
        ready, node_ip
    );
    let channel_clone = channel.clone();
    let response = mnccd_grpc::call_with_retry(
        move || {
            let channel = channel_clone.clone();
            let req = tonic::Request::new(proto::MnccdGrpcSetNvleReadyRequest { ready });
            Box::pin(async move {
                let mut client = MnccdGrpcClient::new(channel);
                client.mnccd_grpc_set_nvle_ready(req).await
            })
        },
        "mnccd_grpc_set_nvle_ready",
    )
    .await;

    match response {
        Ok(resp) => {
            let response_inner = resp.into_inner();
            let ready_uuids: HashSet<String> = response_inner.ready_gpu_uuids.into_iter().collect();
            let missing = sorted_set_difference(&expected_uuids, &ready_uuids);
            let unexpected = sorted_set_difference(&ready_uuids, &expected_uuids);
            if !missing.is_empty() || !unexpected.is_empty() {
                return Err(format!(
                    "Node {} returned a GPU set that does not match the expected topology while setting NVLE ready state to {}. Missing: [{}]. Unexpected: [{}]",
                    node_ip,
                    ready,
                    missing.join(", "),
                    unexpected.join(", ")
                ));
            }
            info!(
                "NVLE ready state set to {} for {} GPU(s) on node {}",
                ready,
                ready_uuids.len(),
                node_ip
            );
            Ok(node_ip)
        }
        Err(e) => Err(format!(
            "Failed to set NVLE ready state to {} on {}: {}",
            ready, node_ip, e
        )),
    }
}

async fn set_nvle_ready_on_nodes(
    channel_map: &HashMap<String, Channel>,
    expected_by_node: &HashMap<String, HashSet<String>>,
    node_ips: &[String],
    ready: bool,
    log_prefix: &str,
) -> (Vec<String>, Vec<String>) {
    let mut ready_tasks = Vec::new();
    let mut all_errors: Vec<String> = Vec::new();
    for node_ip in node_ips {
        let node_ip = node_ip.clone();
        let channel = match channel_map.get(&node_ip) {
            Some(channel) => channel.clone(),
            None => {
                all_errors.push(format!("Channel not found for node {}", node_ip));
                continue;
            }
        };
        let expected_uuids = expected_by_node.get(&node_ip).cloned().unwrap_or_default();
        ready_tasks.push(set_nvle_ready_on_node(
            channel,
            node_ip,
            expected_uuids,
            ready,
            log_prefix.to_string(),
        ));
    }

    let mut successful_nodes = Vec::new();
    for result in join_all(ready_tasks).await {
        match result {
            Ok(node_ip) => successful_nodes.push(node_ip),
            Err(e) => all_errors.push(e),
        }
    }

    (successful_nodes, all_errors)
}

async fn set_nvle_ready_on_all_nodes(
    channel_map: &HashMap<String, Channel>,
    gpu_infos: &[mnccd_grpc::proto::MnccdGrpcGpuInfo],
    ready: bool,
    log_prefix: &str,
    rollback_ready_on_error: Option<bool>,
) -> Result<(), String> {
    info!(
        "{log_prefix}: Setting NVLE ready state to {} on all GPUs on every node...",
        ready
    );

    let mut expected_by_node: HashMap<String, HashSet<String>> = HashMap::new();
    for gpu in gpu_infos {
        expected_by_node
            .entry(gpu.node_ip.clone())
            .or_default()
            .insert(gpu.uuid.clone());
    }

    let node_ips = CONFIG.node_ips.clone();
    let (successful_nodes, mut all_errors) =
        set_nvle_ready_on_nodes(channel_map, &expected_by_node, &node_ips, ready, log_prefix).await;

    if !all_errors.is_empty() {
        if let Some(rollback_ready) = rollback_ready_on_error {
            if !successful_nodes.is_empty() {
                warn!(
                    "{log_prefix}: NVLE ready state update to {} failed on some nodes; restoring {} successful node(s) to {}",
                    ready,
                    successful_nodes.len(),
                    rollback_ready
                );
                let rollback_prefix = format!("{log_prefix} rollback");
                let (_, rollback_errors) = set_nvle_ready_on_nodes(
                    channel_map,
                    &expected_by_node,
                    &successful_nodes,
                    rollback_ready,
                    &rollback_prefix,
                )
                .await;
                if !rollback_errors.is_empty() {
                    all_errors.push(format!(
                        "Best-effort NVLE ready-state reconcile to {} failed: {}",
                        rollback_ready,
                        rollback_errors.join("; ")
                    ));
                }
            }
        }
        return Err(format!(
            "Failed to set NVLE ready state to {} on some nodes: {}",
            ready,
            all_errors.join("; ")
        ));
    }

    Ok(())
}

/// Refreshes NVLE keys for an observed DRAIN_P2P recovery event.
///
/// If another caller handled the event while this call was waiting for the
/// refresh lock, this returns success without re-triggering DRAIN_P2P.
pub async fn refresh_nvle_keys_for_drain_p2p_event(
    trigger_gpu_uuids: Vec<String>,
) -> Result<(), String> {
    if !utils::is_leader()? {
        info!("Not the leader node, skipping NVLE key refresh");
        return Ok(());
    }

    let trigger_gpu_uuids: HashSet<String> = trigger_gpu_uuids.into_iter().collect();
    if trigger_gpu_uuids.is_empty() {
        info!("Skipping NVLE key refresh for empty DRAIN_P2P event");
        return Ok(());
    }

    let _refresh_guard = NVLE_KEY_REFRESH_LOCK.lock().await;

    info!("Starting NVLE key refresh on all GPU pairs...");
    let channel_map = create_channel_map_for_configured_nodes().await?;

    let drained_gpu_uuids =
        collect_drain_p2p_recovery_gpu_uuids_from_all_nodes(&channel_map).await?;
    if trigger_gpu_uuids.is_disjoint(&drained_gpu_uuids) {
        info!(
            "Skipping stale NVLE key refresh event; none of {} trigger GPU(s) remain in DRAIN_P2P",
            trigger_gpu_uuids.len()
        );
        return Ok(());
    }

    info!(
        "Handling NVLE key refresh for {} GPU(s) currently in DRAIN_P2P",
        drained_gpu_uuids.len()
    );

    // Remap tables are locked after initial setup, so refresh must use the
    // ALID map validated and cached before lock instead of querying again.
    let setup_context = get_cached_nvle_setup_context()?;
    let cached_gpu_uuids: HashSet<String> = setup_context
        .gpu_infos
        .iter()
        .map(|gpu| gpu.uuid.clone())
        .collect();
    let uncached_drained_uuids = sorted_set_difference(&drained_gpu_uuids, &cached_gpu_uuids);
    if !uncached_drained_uuids.is_empty() {
        return Err(format!(
            "DRAIN_P2P reported for GPU(s) not present in cached NVLE setup context: {}",
            uncached_drained_uuids.join(", ")
        ));
    }

    let refresh_jobs = build_nvle_key_setup_jobs(
        &setup_context.gpu_infos,
        &setup_context.uuid_to_alid,
        &channel_map,
    )?;
    if refresh_jobs.is_empty() {
        info!("Skipping NVLE key refresh; fewer than two GPUs found");
        return Ok(());
    }

    set_nvle_ready_on_all_nodes(
        &channel_map,
        &setup_context.gpu_infos,
        false,
        "Refresh",
        Some(true),
    )
    .await?;

    info!("Refreshing encryption keys for all GPU pairs; retrying while RM reports busy...");
    if let Err(refresh_error) = program_nvle_key_refresh_jobs(refresh_jobs).await {
        warn!(
            "NVLE key refresh failed after ready=false; attempting to restore NVLE ready state to true"
        );
        if let Err(restore_error) = set_nvle_ready_on_all_nodes(
            &channel_map,
            &setup_context.gpu_infos,
            true,
            "Refresh restore",
            None,
        )
        .await
        {
            return Err(format!(
                "NVLE key refresh failed: {}; additionally failed to restore NVLE ready state to true: {}",
                refresh_error, restore_error
            ));
        }
        return Err(refresh_error);
    }

    set_nvle_ready_on_all_nodes(
        &channel_map,
        &setup_context.gpu_infos,
        true,
        "Refresh",
        Some(false),
    )
    .await?;
    info!("NVLE key refresh completed successfully");

    Ok(())
}

/// Returns `true` only if every unordered pair of GPUs has the same remap table shape and
/// identical `fla_remap_table_addr` and `gpa_remap_table_addr` entries at each index.
/// Returns `false` if any pair disagrees on table size, vector length, or any address value.
/// Empty or single-GPU input is treated as valid.
pub fn validate_remap_tables_on_all_gpus(per_gpu: &[MnccdGrpcPerGpuRemapTableInfo]) -> bool {
    for i in 0..per_gpu.len() {
        for j in (i + 1)..per_gpu.len() {
            let a = &per_gpu[i];
            let b = &per_gpu[j];
            if a.remap_tab_size != b.remap_tab_size {
                return false;
            }
            if a.fla_remap_table_addr.len() != a.gpa_remap_table_addr.len()
                || b.fla_remap_table_addr.len() != b.gpa_remap_table_addr.len()
            {
                return false;
            }
            if a.fla_remap_table_addr.len() != b.fla_remap_table_addr.len() {
                return false;
            }
            for k in 0..a.fla_remap_table_addr.len() {
                if a.fla_remap_table_addr[k] != b.fla_remap_table_addr[k]
                    || a.gpa_remap_table_addr[k] != b.gpa_remap_table_addr[k]
                {
                    return false;
                }
            }
        }
    }
    true
}

async fn setup_nvle_encryption_key_rpc(
    channel: Channel,
    device_uuid: String,
    local_gpu_alid: u32,
    remote_gpu_alid: u32,
    local_gpu_nvle_private_info: Vec<u8>,
    remote_gpu_nvle_private_info: Vec<u8>,
    nvle_key: Vec<u8>,
) -> Result<
    tonic::Response<mnccd_grpc::proto::MnccdGrpcSetupNvleEncryptionKeyResponse>,
    tonic::Status,
> {
    let channel_clone = channel.clone();
    let req_data = mnccd_grpc::proto::MnccdGrpcSetupNvleEncryptionKeyRequest {
        device_uuid,
        local_gpu_alid,
        remote_gpu_alid,
        local_gpu_nvle_private_info,
        remote_gpu_nvle_private_info,
        nvle_key,
    };
    mnccd_grpc::call_with_retry(
        move || {
            let channel = channel_clone.clone();
            let req = tonic::Request::new(req_data.clone());
            Box::pin(async move {
                let mut client = MnccdGrpcClient::new(channel);
                client.mnccd_grpc_setup_nvle_encryption_key(req).await
            })
        },
        "mnccd_grpc_setup_nvle_encryption_key",
    )
    .await
}

/// Setup NVLE on all GPUs using gRPC calls
///
/// This function performs the complete NVLE setup process using gRPC:
/// 1. Confirm fabric probe completion and collect GPU information from all nodes
/// 2. Query remap tables on all nodes and obtain each GPU's ALID
/// 3. Validate remap table coverage and consistency across all GPUs
/// 4. Lock remap table and MSE on all nodes
/// 5. Setup encryption keys for all GPU pairs
/// 6. Set NVLE ready state to true on all nodes
///
/// # Returns
///
/// Returns `Ok(())` if setup succeeds, or an error message if any step fails
pub async fn setup_nvle_on_all_gpus() -> Result<(), String> {
    use mnccd_grpc::proto;

    // Only the leader should perform NVLE setup (uses the same [`CONFIG`] as main).
    if !utils::is_leader()? {
        info!("Not the leader node, skipping NVLE setup");
        return Ok(());
    }

    let _setup_guard = NVLE_KEY_REFRESH_LOCK.lock().await;

    info!("Starting NVLE setup on all GPUs on every node...");

    // One channel per node (created concurrently), reused across all setup steps.
    let channel_map = create_channel_map_for_configured_nodes().await?;

    // Confirm fabric probe completion and collect GPU information from all nodes.
    let gpu_infos = collect_gpu_infos_from_all_nodes(&channel_map, "Initial Nvle Setup").await?;

    // Query remap table on all nodes (concurrent per node) and consume each GPU's ALID.
    let all_per_gpu_remap_table =
        collect_remap_table_info_from_all_nodes(&channel_map, "Initial Nvle Setup").await?;

    // ALID for each GPU is sourced from the remap table query results.
    validate_remap_table_coverage(&gpu_infos, &all_per_gpu_remap_table)?;
    let uuid_to_alid = build_uuid_to_alid_map(&all_per_gpu_remap_table)?;

    // Validate remap tables match across all GPUs (pairwise FLA/GPA)
    info!("Initial Nvle Setup: Validating remap table consistency across all GPUs...");
    if !validate_remap_tables_on_all_gpus(&all_per_gpu_remap_table) {
        return Err(
            "Remap table validation failed: not all GPUs have matching FLA and GPA remap table addresses"
                .to_string(),
        );
    }
    info!(
        "Initial Nvle Setup: Remap table consistency validated (pairwise FLA/GPA addresses match for all GPUs)"
    );

    // Lock remap table and MSE on all nodes (concurrent per node)
    info!("Initial Nvle Setup: Locking remap table and MSE of all GPUs on every node...");
    {
        let mut lock_tasks = Vec::new();
        let mut all_errors: Vec<String> = Vec::new();
        for node_ip in &CONFIG.node_ips {
            let node_ip = node_ip.clone();
            let channel = channel_map
                .get(&node_ip)
                .ok_or_else(|| format!("Channel not found for node {}", node_ip))?
                .clone();

            lock_tasks.push(async move {
                debug!("Locking remap table on node {}...", node_ip);
                let channel_clone = channel.clone();
                let response = mnccd_grpc::call_with_retry(
                    move || {
                        let channel = channel_clone.clone();
                        let req = tonic::Request::new(proto::MnccdGrpcLockRemapTableRequest {});
                        Box::pin(async move {
                            let mut client = MnccdGrpcClient::new(channel);
                            client.mnccd_grpc_lock_remap_table(req).await
                        })
                    },
                    "mnccd_grpc_lock_remap_table",
                )
                .await;

                match response {
                    Ok(resp) => {
                        let response_inner = resp.into_inner();
                        if !response_inner.success {
                            Err(format!(
                                "Failed to lock remap table on {}: {}",
                                node_ip, response_inner.error_message
                            ))
                        } else {
                            info!(
                                "Remap table locked for {} GPUs on node {}",
                                response_inner.locked_gpu_uuids.len(),
                                node_ip
                            );
                            Ok(())
                        }
                    }
                    Err(e) => Err(format!("Failed to lock remap table on {}: {}", node_ip, e)),
                }
            });
        }

        for result in join_all(lock_tasks).await {
            if let Err(e) = result {
                all_errors.push(e);
            }
        }

        if !all_errors.is_empty() {
            return Err(format!(
                "Failed to lock remap table on some nodes: {}",
                all_errors.join("; ")
            ));
        }
    }

    // Setup encryption keys for all GPU pairs
    info!("Initial Nvle Setup: Setting up encryption keys for all GPU pairs...");
    {
        let mut errors: Vec<String> = Vec::new();
        for i in 0..gpu_infos.len() {
            for j in (i + 1)..gpu_infos.len() {
                let local_gpu = &gpu_infos[i];
                let remote_gpu = &gpu_infos[j];

                let key_secret = generate_random_bytes(32).map_err(|e| {
                    format!(
                        "Failed to generate encryption key secret for GPU pair {} <-> {}: {}",
                        local_gpu.uuid, remote_gpu.uuid, e
                    )
                })?;

                // Opaque NVLE private-info blobs collected per GPU (fed verbatim to key setup).
                let local_nvle_private_info = local_gpu.nvle_private_info.clone();
                let remote_nvle_private_info = remote_gpu.nvle_private_info.clone();

                // Cross-node pairs: both RPCs in parallel. Same-node pairs: serialized (NVML).
                let local_uuid = local_gpu.uuid.clone();
                let remote_uuid = remote_gpu.uuid.clone();
                let local_node_ip = local_gpu.node_ip.clone();
                let remote_node_ip = remote_gpu.node_ip.clone();
                let local_alid = *uuid_to_alid
                    .get(&local_gpu.uuid)
                    .ok_or_else(|| format!("ALID not found for GPU {}", local_gpu.uuid))?;
                let remote_alid = *uuid_to_alid
                    .get(&remote_gpu.uuid)
                    .ok_or_else(|| format!("ALID not found for GPU {}", remote_gpu.uuid))?;
                let same_node = local_node_ip == remote_node_ip;

                let local_channel = channel_map
                    .get(&local_node_ip)
                    .ok_or_else(|| {
                        format!(
                            "Channel not found for node {} (GPU {})",
                            local_node_ip, local_uuid
                        )
                    })?
                    .clone();
                let remote_channel = channel_map
                    .get(&remote_node_ip)
                    .ok_or_else(|| {
                        format!(
                            "Channel not found for node {} (GPU {})",
                            remote_node_ip, remote_uuid
                        )
                    })?
                    .clone();

                let key_secret_for_local = key_secret.clone();
                let key_secret_for_remote = key_secret;
                let local_nvle_private_info_for_local = local_nvle_private_info.clone();
                let remote_nvle_private_info_for_local = remote_nvle_private_info.clone();
                let local_nvle_private_info_for_remote = local_nvle_private_info;
                let remote_nvle_private_info_for_remote = remote_nvle_private_info;

                let (local_response, remote_response) = if same_node {
                    (
                        setup_nvle_encryption_key_rpc(
                            local_channel,
                            local_uuid,
                            local_alid,
                            remote_alid,
                            local_nvle_private_info_for_local,
                            remote_nvle_private_info_for_local,
                            key_secret_for_local,
                        )
                        .await,
                        setup_nvle_encryption_key_rpc(
                            remote_channel,
                            remote_uuid,
                            remote_alid,
                            local_alid,
                            remote_nvle_private_info_for_remote,
                            local_nvle_private_info_for_remote,
                            key_secret_for_remote,
                        )
                        .await,
                    )
                } else {
                    tokio::join!(
                        setup_nvle_encryption_key_rpc(
                            local_channel,
                            local_uuid,
                            local_alid,
                            remote_alid,
                            local_nvle_private_info_for_local,
                            remote_nvle_private_info_for_local,
                            key_secret_for_local,
                        ),
                        setup_nvle_encryption_key_rpc(
                            remote_channel,
                            remote_uuid,
                            remote_alid,
                            local_alid,
                            remote_nvle_private_info_for_remote,
                            local_nvle_private_info_for_remote,
                            key_secret_for_remote,
                        ),
                    )
                };

                match local_response {
                    Ok(resp) => {
                        let inner = resp.into_inner();
                        if !inner.success {
                            errors.push(format!(
                                "Failed to setup encryption key from {} (node {}) to {} (node {}): {}",
                                local_gpu.uuid,
                                local_gpu.node_ip,
                                remote_gpu.uuid,
                                remote_gpu.node_ip,
                                inner.error_message
                            ));
                        }
                    }
                    Err(e) => {
                        errors.push(format!(
                            "Failed to setup encryption key from {} (node {}) to {} (node {}): {}",
                            local_gpu.uuid,
                            local_gpu.node_ip,
                            remote_gpu.uuid,
                            remote_gpu.node_ip,
                            e
                        ));
                    }
                }

                match remote_response {
                    Ok(resp) => {
                        let inner = resp.into_inner();
                        if !inner.success {
                            errors.push(format!(
                                "Failed to setup encryption key from {} (node {}) to {} (node {}): {}",
                                remote_gpu.uuid,
                                remote_gpu.node_ip,
                                local_gpu.uuid,
                                local_gpu.node_ip,
                                inner.error_message
                            ));
                        }
                    }
                    Err(e) => {
                        errors.push(format!(
                            "Failed to setup encryption key from {} (node {}) to {} (node {}): {}",
                            remote_gpu.uuid,
                            remote_gpu.node_ip,
                            local_gpu.uuid,
                            local_gpu.node_ip,
                            e
                        ));
                    }
                }
            }
        }

        if !errors.is_empty() {
            return Err(format!(
                "Encryption key setup completed with errors: {}",
                errors.join("; ")
            ));
        }
        info!("Encryption keys setup complete for all GPU pairs");
    }

    set_nvle_ready_on_all_nodes(
        &channel_map,
        &gpu_infos,
        true,
        "Initial Nvle Setup",
        Some(false),
    )
    .await?;

    cache_nvle_setup_context(NvleSetupContext {
        gpu_infos: gpu_infos.clone(),
        uuid_to_alid: uuid_to_alid.clone(),
    })?;

    info!("NVLE setup completed successfully!");
    Ok(())
}

#[cfg(test)]
mod nvle_setup_context_tests {
    use super::{cache_nvle_setup_context, get_cached_nvle_setup_context, NvleSetupContext};
    use crate::mnccd_grpc::proto::MnccdGrpcGpuInfo;
    use std::collections::HashMap;

    #[test]
    fn cached_context_round_trips_gpu_info_and_alids() {
        let gpu_infos = vec![MnccdGrpcGpuInfo {
            uuid: "GPU-test".to_string(),
            nvle_private_info: vec![1, 2, 3, 4],
            node_ip: "10.0.0.1".to_string(),
        }];
        let uuid_to_alid = HashMap::from([("GPU-test".to_string(), 7)]);

        cache_nvle_setup_context(NvleSetupContext {
            gpu_infos: gpu_infos.clone(),
            uuid_to_alid: uuid_to_alid.clone(),
        })
        .expect("cache setup context");

        let cached = get_cached_nvle_setup_context().expect("read setup context");
        assert_eq!(cached.gpu_infos.len(), 1);
        assert_eq!(cached.gpu_infos[0].uuid, "GPU-test");
        assert_eq!(cached.gpu_infos[0].nvle_private_info, vec![1, 2, 3, 4]);
        assert_eq!(cached.gpu_infos[0].node_ip, "10.0.0.1");
        assert_eq!(cached.uuid_to_alid.get("GPU-test"), Some(&7));
    }
}

#[cfg(test)]
mod logging_filter_tests {
    use super::{Level, CRATE_NAME};
    use tracing_subscriber::filter::EnvFilter;

    /// `tracing::Level`'s `Display` is uppercase (e.g. `DEBUG`). The default
    /// filter MNCCD builds embeds it in a directive; if `EnvFilter` didn't accept
    /// that casing it would silently drop the crate-scoped directive and stop
    /// limiting verbosity to MNCCD. Parse strictly to catch that regression.
    #[test]
    fn default_filter_directive_parses_for_every_level() {
        for level in [
            Level::ERROR,
            Level::WARN,
            Level::INFO,
            Level::DEBUG,
            Level::TRACE,
        ] {
            let directive = format!("info,{CRATE_NAME}={level}");
            EnvFilter::builder()
                .parse(&directive)
                .unwrap_or_else(|e| panic!("directive {directive:?} must parse: {e}"));
        }
    }
}

#[cfg(test)]
mod endpoint_url_tests {
    use super::{
        build_client_endpoint_url, format_host_for_uri_authority, leader_ip_from_node_ips,
        RawConfig,
    };
    use std::net::{IpAddr, SocketAddr};

    #[test]
    fn ipv4_host_unchanged() {
        assert_eq!(
            format_host_for_uri_authority("10.0.0.1"),
            "10.0.0.1"
        );
    }

    #[test]
    fn ipv6_host_is_bracketed() {
        assert_eq!(
            format_host_for_uri_authority("2001:db8::1"),
            "[2001:db8::1]"
        );
    }

    #[test]
    fn build_url_brackets_ipv6() {
        assert_eq!(
            build_client_endpoint_url("https", "2001:db8::1", 50051),
            "https://[2001:db8::1]:50051"
        );
    }

    #[test]
    fn build_url_leaves_ipv4_unbracketed() {
        assert_eq!(
            build_client_endpoint_url("https", "10.0.0.1", 50051),
            "https://10.0.0.1:50051"
        );
    }

    #[test]
    fn socket_addr_display_brackets_ipv6() {
        let ip: IpAddr = "2001:db8::1".parse().unwrap();
        let addr = SocketAddr::new(ip, 50051);
        assert_eq!(addr.to_string(), "[2001:db8::1]:50051");
    }

    #[test]
    fn socket_addr_display_ipv4() {
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        let addr = SocketAddr::new(ip, 50051);
        assert_eq!(addr.to_string(), "10.0.0.1:50051");
    }

    #[test]
    fn node_ips_ipv6_build_bracketed_endpoint_urls() {
        // Scenario: mnccd_config.toml `node_ips` field contains IPv6 addresses.
        let config_toml = r#"
[server]
server_port = 50051
[cluster]
node_ips = ["2001:db8::1", "fe80::1", "2001:db8::2"]
[retry_policy]
max_retries = 1
initial_backoff_ms = 1
max_backoff_ms = 1
"#;
        let raw: RawConfig = toml::from_str(config_toml).unwrap();

        // Every IPv6 node IP must be bracketed when placed into a gRPC endpoint URL.
        for node_ip in &raw.cluster.node_ips {
            assert_eq!(
                format_host_for_uri_authority(node_ip),
                format!("[{node_ip}]")
            );
            assert_eq!(
                build_client_endpoint_url("https", node_ip, 50051),
                format!("https://[{node_ip}]:50051")
            );
        }

        // The leader (lowest address) derived from IPv6 node_ips is also bracketed.
        let leader_ip = leader_ip_from_node_ips(&raw.cluster.node_ips).unwrap();
        assert_eq!(leader_ip, "2001:db8::1");
        assert_eq!(
            build_client_endpoint_url("https", &leader_ip, 50051),
            "https://[2001:db8::1]:50051"
        );
    }

    #[test]
    fn node_ips_mixed_ipv4_and_ipv6_build_correct_endpoint_urls() {
        // Scenario: mnccd_config.toml `node_ips` field mixes IPv4 and IPv6 addresses.
        let config_toml = r#"
[server]
server_port = 50051
[cluster]
node_ips = ["2001:db8::2", "10.0.0.5", "fe80::1", "10.0.0.2"]
[retry_policy]
max_retries = 1
initial_backoff_ms = 1
max_backoff_ms = 1
"#;
        let raw: RawConfig = toml::from_str(config_toml).unwrap();

        // Each entry is formatted independently: IPv6 literals are bracketed, IPv4 are not.
        let expected = [
            ("2001:db8::2", "https://[2001:db8::2]:50051"),
            ("10.0.0.5", "https://10.0.0.5:50051"),
            ("fe80::1", "https://[fe80::1]:50051"),
            ("10.0.0.2", "https://10.0.0.2:50051"),
        ];
        for ((node_ip, expected_url), raw_ip) in expected.iter().zip(&raw.cluster.node_ips) {
            assert_eq!(node_ip, raw_ip);
            assert_eq!(
                build_client_endpoint_url("https", node_ip, 50051),
                *expected_url
            );
        }

        // `IpAddr` orders every IPv4 address before every IPv6 address (comparison is by
        // enum variant, not numeric), so whenever the list contains any IPv4 address the
        // leader is the lowest IPv4 address regardless of the IPv6 entries.
        let leader_ip = leader_ip_from_node_ips(&raw.cluster.node_ips).unwrap();
        assert_eq!(leader_ip, "10.0.0.2");
        assert_eq!(
            build_client_endpoint_url("https", &leader_ip, 50051),
            "https://10.0.0.2:50051"
        );
    }
}

#[cfg(test)]
mod config_tests {
    use super::RawConfig;

    const MIN_CONFIG: &str = r#"
[server]
server_port = 50051
[cluster]
node_ips = ["10.0.0.1"]
[retry_policy]
max_retries = 1
initial_backoff_ms = 1
max_backoff_ms = 1
"#;

    #[test]
    fn empty_psk_key_allowed_when_not_psk_mode() {
        let raw: RawConfig = toml::from_str(&format!(
            "{MIN_CONFIG}\n[tls]\nmode = \"none\"\npsk_key = \"\"\n"
        ))
        .unwrap();
        let mode = super::parse_grpc_tls_mode(&raw.tls.mode).unwrap();
        assert_eq!(mode, super::GrpcTlsMode::Plain);
        assert!(super::psk_tls::PskTlsConfig::from_strings(
            &raw.tls.psk_identity,
            &raw.tls.psk_key
        )
        .is_err());
    }

    #[test]
    fn psk_fields_default_empty_when_tls_section_omitted() {
        let raw: RawConfig = toml::from_str(MIN_CONFIG).unwrap();
        assert_eq!(raw.tls.mode, "psk");
        assert!(raw.tls.psk_identity.is_empty());
        assert!(raw.tls.psk_key.is_empty());
    }

    #[test]
    fn psk_mode_rejects_empty_identity() {
        let err = super::psk_tls::PskTlsConfig::from_strings("", "some-key").unwrap_err();
        assert!(
            err.contains("psk_identity"),
            "expected empty identity error, got: {err}"
        );
    }

    #[test]
    fn psk_mode_rejects_empty_key() {
        let err = super::psk_tls::PskTlsConfig::from_strings("some-identity", "").unwrap_err();
        assert!(
            err.contains("psk_key"),
            "expected empty key error, got: {err}"
        );
    }

    #[test]
    fn tls_mode_defaults_to_psk() {
        let raw: RawConfig = toml::from_str(MIN_CONFIG).unwrap();
        assert_eq!(raw.tls.mode, "psk");
    }

    #[test]
    fn parse_tls_mode_values() {
        assert_eq!(
            super::parse_grpc_tls_mode("psk").unwrap(),
            super::GrpcTlsMode::Psk
        );
        assert_eq!(
            super::parse_grpc_tls_mode("none").unwrap(),
            super::GrpcTlsMode::Plain
        );
        assert_eq!(
            super::parse_grpc_tls_mode("mtls").unwrap(),
            super::GrpcTlsMode::Mtls
        );
        for invalid in ["invalid", "plain", "cert", "certs"] {
            assert!(
                super::parse_grpc_tls_mode(invalid).is_err(),
                "expected {invalid:?} to be rejected"
            );
        }
    }

    #[test]
    fn peer_cert_fields_default_empty_when_tls_section_omitted() {
        let raw: RawConfig = toml::from_str(MIN_CONFIG).unwrap();
        assert!(raw.tls.server_name.is_empty());
        assert!(raw.tls.allowed_client_organizations.is_empty());
    }

    #[test]
    fn peer_cert_fields_parse_from_tls_section() {
        let raw: RawConfig = toml::from_str(&format!(
            "{MIN_CONFIG}\n[tls]\nmode = \"mtls\"\nserver_name = \"group-a.example.com\"\n\
             allowed_client_organizations = [\"example-group-a\", \"example-group-b\"]\n"
        ))
        .unwrap();
        assert_eq!(raw.tls.server_name, "group-a.example.com");
        assert_eq!(
            raw.tls.allowed_client_organizations,
            ["example-group-a", "example-group-b"]
        );
    }
}
