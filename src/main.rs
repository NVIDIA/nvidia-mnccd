/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use clap::Parser;
use futures::future::join_all;
use std::error::Error;
use std::io::IsTerminal;
use tracing::{error, info, warn};

/// Minimum severity of log records to emit.
///
/// A thin `clap::ValueEnum` wrapper so `--log-level` gets validation and
/// `[possible values: ...]` in `--help`; it converts to [`tracing::Level`],
/// which itself does not implement `ValueEnum`.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl From<LogLevel> for tracing::Level {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Error => tracing::Level::ERROR,
            LogLevel::Warn => tracing::Level::WARN,
            LogLevel::Info => tracing::Level::INFO,
            LogLevel::Debug => tracing::Level::DEBUG,
            LogLevel::Trace => tracing::Level::TRACE,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "nvidia-mnccd",
    version,
    about = "Nvidia Multi-Node Confidential Compute Daemon",
    after_long_help = "Configuration:\n  mnccd_config.toml is loaded from the first of: $MNCCD_DATA_DIR, $CARGO_MANIFEST_DIR (set by `cargo run`), or the compile-time default (overridable at build time via $MNCCD_DEFAULT_DATA_DIR).\n\nLogging:\n  --log-level sets the base verbosity. The RUST_LOG environment variable, if set, overrides it and supports per-module filters (e.g. RUST_LOG=info,nvidia_mnccd=debug)."
)]
struct Cli {
    /// Skip NVLE setup on the leader node after peer connectivity checks.
    #[arg(long = "skip-nvle")]
    skip_nvle: bool,
    /// Disable TLS for gRPC server and client connections.
    /// Strictly for development platforms only; not recommended on production.
    #[arg(long = "no-tls")]
    no_tls: bool,
    /// Detach from the terminal; log stdout/stderr to /tmp/nvidia-mnccd.{out,err}.
    #[arg(long = "daemonize")]
    daemonize: bool,
    /// Base log verbosity (overridden by the RUST_LOG environment variable).
    #[arg(long = "log-level", value_enum, default_value_t = LogLevel::Info)]
    log_level: LogLevel,
}

/// Rejects the one unsupported privilege/mode combination: non-root in the foreground.
///
/// Foreground runs take the single-instance flock in `/run/nvidia-mnccd`, which only root
/// may create or open. A per-user lock under `/run/user/<uid>` is not an option either:
/// an instance started by a different user would not see it, so it would not actually
/// enforce one instance per host. `--daemonize` locks a PID file under `/tmp` instead and
/// works for any user.
#[cfg(unix)]
fn check_privileges(daemonize: bool) -> Result<(), Box<dyn Error>> {
    // SAFETY: `libc::geteuid` is an `unsafe fn` (FFI); it returns the effective UID only.
    let euid = unsafe { libc::geteuid() };
    if euid == 0 {
        return Ok(());
    }

    if !daemonize {
        return Err(format!(
            "not running as root (effective UID {euid}): foreground mode requires root because \
             the single-instance lock lives in /run/nvidia-mnccd; re-run as root or pass --daemonize"
        )
        .into());
    }

    warn!("not running as root (effective UID {euid}); GPU or privileged operations may fail");
    Ok(())
}

#[cfg(not(unix))]
fn check_privileges(_daemonize: bool) -> Result<(), Box<dyn Error>> {
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    // Disable ANSI when detaching (logs go to a file) or when stdout isn't a TTY,
    // so control characters don't pollute captured/redirected output.
    let enable_ansi = !cli.daemonize && std::io::stdout().is_terminal();
    nvidia_mnccd::init_logging(cli.log_level.into(), enable_ansi);

    if let Err(e) = check_privileges(cli.daemonize) {
        error!("{e}");
        return Err(e);
    }

    if let Err(e) = nvidia_mnccd::init_config(cli.no_tls) {
        error!("failed to load configuration: {e}");
        return Err(e.into());
    }

    if cli.daemonize {
        nvidia_mnccd::start_daemon()?;
    } else {
        let _instance_lock = nvidia_mnccd::acquire_instance_lock()?;
    }

    let tls_mode = nvidia_mnccd::CONFIG.grpc_tls_mode;
    nvidia_mnccd::warn_if_plaintext_grpc(tls_mode);
    nvidia_mnccd::init_grpc_tls_mode(tls_mode);

    tokio_main(cli.skip_nvle)
}

#[tokio::main]
async fn tokio_main(skip_nvle: bool) -> Result<(), Box<dyn Error>> {
    // Server
    let t1 = nvidia_mnccd::spawn_and_start_server()
        .await
        .inspect_err(|e| error!("failed to start gRPC server: {e}"))?;

    // Clients - spawn for each non-local IP in node_ips
    let mut client_handles = Vec::new();
    for node_ip in &nvidia_mnccd::CONFIG.node_ips {
        if !nvidia_mnccd::is_local(node_ip)? {
            let handle = nvidia_mnccd::spawn_and_start_client(node_ip);
            client_handles.push(handle);
        }
    }

    // Wait for all client echo tasks to complete (confirms connectivity to all peers)
    let results = join_all(client_handles).await;
    for result in results {
        result
            .map_err(|join_err| -> Box<dyn Error> { join_err.into() })?
            .inspect_err(|client_err| {
                error!(
                    "Peer connectivity check failed for {}: {client_err}",
                    client_err.node_addr()
                );
            })?;
    }

    // Tests rely on this line. Don't delete!
    info!("Peer connectivity check done!");

    // If this is the leader node, trigger normal NVLE setup across all nodes.
    if nvidia_mnccd::is_leader()? && !skip_nvle {
        nvidia_mnccd::setup_nvle_on_all_gpus()
            .await
            .inspect_err(|e| error!("NVLE setup failed: {e}"))?;
    }

    let _retrain_monitor = if !skip_nvle {
        Some(nvidia_mnccd::spawn_nvle_retrain_monitor())
    } else {
        None
    };

    // Keep the server running; exit if the server task fails or panics.
    t1.await?
        .inspect_err(|e| error!("gRPC server exited: {e}"))
        .map_err(|e| -> Box<dyn Error> { e })?;

    Ok(())
}
