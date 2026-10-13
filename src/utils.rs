/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cluster role and node identity helpers.

use crate::CONFIG;
use local_ip_address::list_afinet_netifas;
use std::collections::HashSet;
use std::net::IpAddr;
use std::sync::OnceLock;

/// Cached result of [`local_node_ip`]. The node's identity is derived from
/// `node_ips` (config, never reloaded at runtime) matched against local
/// interfaces, so it is stable for the process lifetime. Only successful
/// resolutions are cached, allowing a transient early-startup failure (e.g. the
/// interface not yet being up) to be retried.
static LOCAL_NODE_IP: OnceLock<IpAddr> = OnceLock::new();

/// Returns this node's IP address as known to the cluster.
///
/// MNCCD must identify (and bind) itself using the address its peers reach it
/// by — an entry in `node_ips` — not the kernel's default-route source address,
/// which may belong to an unrelated interface on a multi-homed host. This
/// enumerates every local AF_INET/AF_INET6 interface address and returns the
/// first one that also appears in `node_ips`.
///
/// The result is cached after the first successful resolution.
pub fn local_node_ip() -> Result<IpAddr, String> {
    if let Some(ip) = LOCAL_NODE_IP.get() {
        return Ok(*ip);
    }

    let ip = resolve_local_node_ip()?;
    // A concurrent first call may have set it already; the value is identical,
    // so ignore the race and keep whichever landed first.
    let _ = LOCAL_NODE_IP.set(ip);
    Ok(ip)
}

/// Performs the actual interface enumeration and `node_ips` match (uncached).
fn resolve_local_node_ip() -> Result<IpAddr, String> {
    let interfaces = list_afinet_netifas()
        .map_err(|e| format!("unable to enumerate local network interfaces: {e}"))?;
    select_matching_ip(&CONFIG.node_ips, &interfaces)
}

/// Selects the first interface address that also appears in `node_ips`.
///
/// Pure (no syscalls) so it can be unit-tested with synthetic interface lists.
/// Both `node_ips` entries and interface addresses are compared as parsed
/// [`IpAddr`] values, so IPv4 and IPv6 are handled uniformly, differing textual
/// spellings of the same address match, and v4/v6 never cross-match.
fn select_matching_ip(
    node_ips: &[String],
    interfaces: &[(String, IpAddr)],
) -> Result<IpAddr, String> {
    let configured: HashSet<IpAddr> = node_ips
        .iter()
        .map(|s| {
            s.parse::<IpAddr>()
                .map_err(|e| format!("invalid IP in node_ips {s:?}: {e}"))
        })
        .collect::<Result<_, _>>()?;

    for (_name, ip) in interfaces {
        if configured.contains(ip) {
            return Ok(*ip);
        }
    }

    Err(format!(
        "no local interface address matches any entry in node_ips ({:?}); \
         local interface addresses are {:?}",
        node_ips,
        interfaces.iter().map(|(_, ip)| *ip).collect::<Vec<_>>(),
    ))
}

/// Returns `true` when the address string `candidate` denotes `node_ip`.
///
/// `candidate` is parsed to an [`IpAddr`] before comparison so that
/// non-canonical spellings (e.g. `2001:0db8:0000::0001` vs `2001:db8::1`) still
/// compare equal, and v4/v6 never cross-match. `label` is used only to make
/// parse-error messages specific to the caller (e.g. `leader_ip`).
fn ip_str_matches(label: &str, candidate: &str, node_ip: IpAddr) -> Result<bool, String> {
    let parsed = candidate
        .parse::<IpAddr>()
        .map_err(|e| format!("invalid {label} {candidate:?}: {e}"))?;
    Ok(parsed == node_ip)
}

/// Returns `true` when this node's IP matches the cluster leader address.
///
/// The leader is the node with the numerically smallest IP in [`crate::CONFIG`].
pub fn is_leader() -> Result<bool, String> {
    ip_str_matches("leader_ip", &CONFIG.leader_ip, local_node_ip()?)
}

/// Returns `true` when `ip` matches this node's IP address.
pub fn is_local(ip: &str) -> Result<bool, String> {
    ip_str_matches("IP", ip, local_node_ip()?)
}

#[cfg(test)]
mod tests {
    use super::select_matching_ip;
    use std::net::IpAddr;

    /// Builds a synthetic interface list of `(name, IpAddr)` pairs.
    fn ifaces(addrs: &[(&str, &str)]) -> Vec<(String, IpAddr)> {
        addrs
            .iter()
            .map(|(name, ip)| {
                (
                    (*name).to_string(),
                    ip.parse::<IpAddr>().expect("valid test IP"),
                )
            })
            .collect()
    }

    #[test]
    fn selects_ipv4_bridge_nic_matching_node_ips_over_default_route_nic() {
        // On a host with a NAT NIC (whose address is the kernel's default-route
        // source IP and is enumerated first) plus a bridge NIC, selection must
        // pick the bridge NIC because it is the one listed in node_ips — not the
        // default-route NIC.
        let node_ips = vec!["10.0.1.1".to_string()];
        let interfaces = ifaces(&[
            ("eth0", "10.0.0.1"), // NAT NIC / default-route source IP
            ("eth1", "10.0.1.1"), // bridge NIC advertised in node_ips
        ]);

        let selected = select_matching_ip(&node_ips, &interfaces).unwrap();
        assert_eq!(selected, "10.0.1.1".parse::<IpAddr>().unwrap());
        assert!(selected.is_ipv4());
    }

    #[test]
    fn matches_ipv6_global_address_in_node_ips() {
        let node_ips = vec![
            "2001:db8::1".to_string(),
            "2001:db8::2".to_string(),
            "2001:db8::3".to_string(),
        ];
        let interfaces = ifaces(&[("lo", "::1"), ("eth0", "2001:db8::2")]);

        let selected = select_matching_ip(&node_ips, &interfaces).unwrap();
        assert_eq!(selected, "2001:db8::2".parse::<IpAddr>().unwrap());
        assert!(selected.is_ipv6());
    }

    #[test]
    fn matches_ipv6_regardless_of_textual_spelling() {
        // node_ips uses the fully-expanded form; the interface uses the
        // compressed form. Both parse to the same IpAddr, so they must match.
        let node_ips = vec!["2001:0db8:0000:0000:0000:0000:0000:0001".to_string()];
        let interfaces = ifaces(&[("eth0", "2001:db8::1")]);

        let selected = select_matching_ip(&node_ips, &interfaces).unwrap();
        assert_eq!(selected, "2001:db8::1".parse::<IpAddr>().unwrap());
    }

    #[test]
    fn selects_ipv6_entry_from_mixed_v4_v6_node_ips() {
        let node_ips = vec!["10.0.0.5".to_string(), "2001:db8::9".to_string()];
        // Host only carries the IPv6 address (plus loopbacks), not the v4 one.
        let interfaces = ifaces(&[("lo", "127.0.0.1"), ("eth0", "2001:db8::9")]);

        let selected = select_matching_ip(&node_ips, &interfaces).unwrap();
        assert_eq!(selected, "2001:db8::9".parse::<IpAddr>().unwrap());
    }

    #[test]
    fn ipv4_node_ip_does_not_cross_match_ipv6_interface() {
        // A v4 entry must never match a v6 interface address (and there is no
        // other match), so this resolves to an error.
        let node_ips = vec!["10.0.0.5".to_string()];
        let interfaces = ifaces(&[("eth0", "2001:db8::5")]);

        assert!(select_matching_ip(&node_ips, &interfaces).is_err());
    }

    #[test]
    fn errors_when_no_ipv6_interface_matches() {
        let node_ips = vec!["2001:db8::1".to_string(), "2001:db8::2".to_string()];
        let interfaces = ifaces(&[("lo", "::1"), ("eth0", "fe80::abcd")]);

        let err = select_matching_ip(&node_ips, &interfaces).unwrap_err();
        assert!(err.contains("no local interface address matches"));
    }

    #[test]
    fn ip_str_matches_true_for_canonical_ipv4() {
        let node_ip = "10.0.0.1".parse::<IpAddr>().unwrap();
        assert!(super::ip_str_matches("leader_ip", "10.0.0.1", node_ip).unwrap());
    }

    #[test]
    fn ip_str_matches_false_for_different_ipv4() {
        let node_ip = "10.0.0.1".parse::<IpAddr>().unwrap();
        assert!(!super::ip_str_matches("leader_ip", "10.0.2.15", node_ip).unwrap());
    }

    #[test]
    fn ip_str_matches_true_for_noncanonical_ipv6_leader_spelling() {
        // This is the reported failure mode: leader_ip written in expanded form
        // must still match the node's canonically-formatted interface address.
        let node_ip = "2001:db8::1".parse::<IpAddr>().unwrap();
        assert!(super::ip_str_matches(
            "leader_ip",
            "2001:0db8:0000:0000:0000:0000:0000:0001",
            node_ip
        )
        .unwrap());
    }

    #[test]
    fn ip_str_matches_false_for_different_ipv6() {
        let node_ip = "2001:db8::1".parse::<IpAddr>().unwrap();
        assert!(!super::ip_str_matches("leader_ip", "2001:db8::2", node_ip).unwrap());
    }

    #[test]
    fn ip_str_matches_does_not_cross_match_v4_and_v6() {
        let node_ip = "2001:db8::1".parse::<IpAddr>().unwrap();
        assert!(!super::ip_str_matches("IP", "10.0.0.1", node_ip).unwrap());
    }

    #[test]
    fn ip_str_matches_errors_on_malformed_candidate_with_label() {
        let node_ip = "10.0.0.1".parse::<IpAddr>().unwrap();
        let err = super::ip_str_matches("leader_ip", "not-an-ip", node_ip).unwrap_err();
        assert!(err.contains("invalid leader_ip"));
    }

    #[test]
    fn rejects_ipv6_node_ip_with_unparseable_zone_id() {
        // std's IpAddr cannot parse a scoped/zoned form like `fe80::1%eth0`,
        // so such a node_ips entry is reported as invalid rather than silently
        // ignored.
        let node_ips = vec!["fe80::1%eth0".to_string()];
        let interfaces = ifaces(&[("eth0", "fe80::1")]);

        let err = select_matching_ip(&node_ips, &interfaces).unwrap_err();
        assert!(err.contains("invalid IP in node_ips"));
    }
}
