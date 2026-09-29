/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Optional mTLS peer certificate checks, configured under `[tls]` in `mnccd_config.toml`.
//!
//! * `server_name`: DNS name that every peer's server certificate must present as a DNS
//!   SAN. Clients check it with standard TLS hostname verification. Defaults to `mnccd`.
//! * `allowed_client_organizations`: Subject Organization (O) values accepted on peer
//!   client certificates. The server checks them on every request. Empty disables the
//!   check.
//!
//! Both are opt-in: when unset, mTLS behaves exactly as before. They only apply to
//! `tls.mode = "mtls"`: setting either with another `tls.mode` is a startup error, and
//! `--no-tls` turns them off along with TLS.

use crate::GrpcTlsMode;
use openssl::nid::Nid;
use openssl::x509::{X509Ref, X509};
use rustls_pki_types::DnsName;
use std::collections::HashSet;
use std::sync::Arc;
use tonic::service::Interceptor;
use tonic::{Request, Status};
use tracing::warn;

/// Name that peer server certificates must present when `[tls].server_name` is unset.
pub(crate) const DEFAULT_SERVER_NAME: &str = "mnccd";

/// gRPC status message for denied callers, the same as the `node_ips` check. The reason
/// is logged server-side only.
const REQUEST_DENIED: &str = "request denied";

/// Validated `[tls].server_name` and `[tls].allowed_client_organizations`.
///
/// The default value turns both checks off.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct PeerCertPolicy {
    /// `None` keeps [`DEFAULT_SERVER_NAME`].
    server_name: Option<String>,
    /// Empty turns the client certificate Organization check off.
    allowed_client_organizations: Vec<String>,
}

impl PeerCertPolicy {
    /// Validates the raw `[tls]` values, trimming surrounding whitespace. An empty or
    /// whitespace-only `server_name` and an empty `allowed_client_organizations` mean
    /// "not set".
    pub(crate) fn from_config(
        server_name: &str,
        allowed_client_organizations: &[String],
    ) -> Result<Self, String> {
        let server_name = match server_name.trim() {
            "" => None,
            name => {
                // Same syntax check rustls applies to the name it verifies. It rejects IP
                // literals and wildcards.
                DnsName::try_from(name).map_err(|_| {
                    format!(
                        "tls.server_name {name:?} must be a DNS name \
                         (IP addresses and wildcards are not supported)"
                    )
                })?;
                Some(name.to_owned())
            }
        };
        let allowed_client_organizations = allowed_client_organizations
            .iter()
            .map(|organization| organization.trim().to_owned())
            .collect::<Vec<_>>();
        if let Some(index) = allowed_client_organizations
            .iter()
            .position(|organization| organization.is_empty())
        {
            return Err(format!(
                "tls.allowed_client_organizations[{index}] must not be empty"
            ));
        }
        Ok(Self {
            server_name,
            allowed_client_organizations,
        })
    }

    /// DNS name that clients require on peer server certificates.
    pub(crate) fn server_name(&self) -> &str {
        self.server_name.as_deref().unwrap_or(DEFAULT_SERVER_NAME)
    }

    /// Fails if a check is configured but `mode` is not mutual TLS.
    pub(crate) fn ensure_supported(&self, mode: GrpcTlsMode) -> Result<(), String> {
        if mode == GrpcTlsMode::Mtls {
            return Ok(());
        }
        let configured = match (
            self.server_name.is_some(),
            !self.allowed_client_organizations.is_empty(),
        ) {
            (false, false) => return Ok(()),
            (true, false) => "tls.server_name",
            (false, true) => "tls.allowed_client_organizations",
            (true, true) => "tls.server_name and tls.allowed_client_organizations",
        };
        Err(format!(
            "{configured} can only be used with mutual TLS (tls.mode = \"mtls\")"
        ))
    }

    /// Interceptor for the client Organization check, or `None` when it is off or
    /// `mode` is not mutual TLS (only mTLS requests carry client certificates).
    pub(crate) fn client_organization_interceptor(
        &self,
        mode: GrpcTlsMode,
    ) -> Option<ClientOrganizationInterceptor> {
        if mode != GrpcTlsMode::Mtls || self.allowed_client_organizations.is_empty() {
            return None;
        }
        Some(ClientOrganizationInterceptor {
            allowed: Arc::new(self.allowed_client_organizations.iter().cloned().collect()),
        })
    }
}

/// Rejects gRPC requests whose mTLS client certificate has no Subject Organization (O)
/// in `[tls].allowed_client_organizations`. Requests without a readable client
/// certificate are denied.
#[derive(Clone, Debug)]
pub(crate) struct ClientOrganizationInterceptor {
    allowed: Arc<HashSet<String>>,
}

impl Interceptor for ClientOrganizationInterceptor {
    fn call(&mut self, request: Request<()>) -> Result<Request<()>, Status> {
        match check_request_client_organization(&self.allowed, &request) {
            Ok(()) => Ok(request),
            Err(reason) => {
                match request.remote_addr() {
                    Some(addr) => {
                        warn!(ip = %addr.ip(), "rejecting mTLS client certificate: {reason}")
                    }
                    None => warn!("rejecting mTLS client certificate: {reason}"),
                }
                Err(Status::permission_denied(REQUEST_DENIED))
            }
        }
    }
}

/// Checks the leaf client certificate of `request`. The error is the reason to log.
fn check_request_client_organization<T>(
    allowed: &HashSet<String>,
    request: &Request<T>,
) -> Result<(), String> {
    let certs = request
        .peer_certs()
        .ok_or("request has no client certificate")?;
    let leaf = certs.first().ok_or("request has no client certificate")?;
    let cert = X509::from_der(leaf.as_ref())
        .map_err(|e| format!("client certificate could not be parsed: {e}"))?;
    check_client_organization(allowed, &cert)
}

/// Accepts `cert` only if it has a Subject Organization (O) in `allowed` (exact match).
/// The error is the reason to log.
fn check_client_organization(allowed: &HashSet<String>, cert: &X509Ref) -> Result<(), String> {
    let organizations = subject_organizations(cert)?;
    if organizations
        .iter()
        .any(|organization| allowed.contains(organization))
    {
        Ok(())
    } else if organizations.is_empty() {
        Err("client certificate has no Subject Organization (O)".to_string())
    } else {
        Err(format!(
            "client certificate Subject Organization {organizations:?} \
             is not in tls.allowed_client_organizations"
        ))
    }
}

/// Subject Organization (O) values of `cert`.
fn subject_organizations(cert: &X509Ref) -> Result<Vec<String>, String> {
    cert.subject_name()
        .entries_by_nid(Nid::ORGANIZATIONNAME)
        .map(|entry| {
            // `to_string` keeps interior NUL bytes, so a value can't match by truncation.
            entry.data().to_string().map_err(|e| {
                format!("client certificate Subject Organization could not be decoded: {e}")
            })
        })
        .collect()
}

#[cfg(test)]
mod test_pki {
    //! Throwaway ECDSA P-256 CA and leaf certificates for mTLS tests.

    use openssl::asn1::{Asn1Time, Asn1Type};
    use openssl::bn::{BigNum, MsbOption};
    use openssl::ec::{EcGroup, EcKey};
    use openssl::hash::MessageDigest;
    use openssl::nid::Nid;
    use openssl::pkey::{PKey, Private};
    use openssl::x509::extension::{
        BasicConstraints, ExtendedKeyUsage, KeyUsage, SubjectAlternativeName,
    };
    use openssl::x509::{X509Builder, X509Extension, X509Name, X509NameBuilder, X509NameRef, X509};

    /// A leaf certificate and its PKCS#8 key, as PEM.
    pub(crate) struct TestCert {
        pub(crate) cert_pem: Vec<u8>,
        pub(crate) key_pem: Vec<u8>,
    }

    impl TestCert {
        /// The leaf certificate.
        pub(crate) fn x509(&self) -> X509 {
            X509::from_pem(&self.cert_pem).unwrap()
        }
    }

    /// Self-signed CA that issues leaf certificates.
    pub(crate) struct TestCa {
        cert: X509,
        key: PKey<Private>,
    }

    impl TestCa {
        pub(crate) fn new() -> Self {
            let key = new_key();
            let name = subject(&[(Nid::COMMONNAME, "Example Test CA", Asn1Type::UTF8STRING)]);
            let mut builder = cert_builder(&name, &name, &key);
            builder
                .append_extension(BasicConstraints::new().critical().ca().build().unwrap())
                .unwrap();
            builder
                .append_extension(
                    KeyUsage::new()
                        .critical()
                        .key_cert_sign()
                        .crl_sign()
                        .build()
                        .unwrap(),
                )
                .unwrap();
            builder.sign(&key, MessageDigest::sha256()).unwrap();
            Self {
                cert: builder.build(),
                key,
            }
        }

        pub(crate) fn cert_pem(&self) -> Vec<u8> {
            self.cert.to_pem().unwrap()
        }

        /// Server certificate whose only DNS SAN is `dns_name`.
        pub(crate) fn server_cert(&self, dns_name: &str) -> TestCert {
            let name = subject(&[(Nid::COMMONNAME, dns_name, Asn1Type::UTF8STRING)]);
            let server_auth = ExtendedKeyUsage::new().server_auth().build().unwrap();
            self.issue(&name, server_auth, Some(dns_name))
        }

        /// Client certificate with one UTF8String Subject O entry per `organizations` item.
        pub(crate) fn client_cert(&self, organizations: &[&str]) -> TestCert {
            let mut entries = vec![(Nid::COMMONNAME, "example-node", Asn1Type::UTF8STRING)];
            entries.extend(
                organizations.iter().map(|organization| {
                    (Nid::ORGANIZATIONNAME, *organization, Asn1Type::UTF8STRING)
                }),
            );
            self.client_cert_with_subject(&subject(&entries))
        }

        /// Client certificate with the given Subject.
        pub(crate) fn client_cert_with_subject(&self, name: &X509NameRef) -> TestCert {
            let client_auth = ExtendedKeyUsage::new().client_auth().build().unwrap();
            self.issue(name, client_auth, None)
        }

        fn issue(
            &self,
            name: &X509NameRef,
            extended_key_usage: X509Extension,
            dns_san: Option<&str>,
        ) -> TestCert {
            let key = new_key();
            let mut builder = cert_builder(name, self.cert.subject_name(), &key);
            builder
                .append_extension(
                    KeyUsage::new()
                        .critical()
                        .digital_signature()
                        .build()
                        .unwrap(),
                )
                .unwrap();
            builder.append_extension(extended_key_usage).unwrap();
            if let Some(dns_name) = dns_san {
                let san = SubjectAlternativeName::new()
                    .dns(dns_name)
                    .build(&builder.x509v3_context(Some(&self.cert), None))
                    .unwrap();
                builder.append_extension(san).unwrap();
            }
            builder.sign(&self.key, MessageDigest::sha256()).unwrap();
            TestCert {
                cert_pem: builder.build().to_pem().unwrap(),
                key_pem: key.private_key_to_pem_pkcs8().unwrap(),
            }
        }
    }

    /// X.509 name built from `(field, value, ASN.1 string type)` entries.
    pub(crate) fn subject(entries: &[(Nid, &str, Asn1Type)]) -> X509Name {
        let mut builder = X509NameBuilder::new().unwrap();
        for &(field, value, string_type) in entries {
            builder
                .append_entry_by_nid_with_type(field, value, string_type)
                .unwrap();
        }
        builder.build()
    }

    fn new_key() -> PKey<Private> {
        let group = EcGroup::from_curve_name(Nid::X9_62_PRIME256V1).unwrap();
        PKey::from_ec_key(EcKey::generate(&group).unwrap()).unwrap()
    }

    /// X.509 v3 builder valid from now for one day, with a random serial number.
    fn cert_builder(name: &X509NameRef, issuer: &X509NameRef, key: &PKey<Private>) -> X509Builder {
        let mut serial = BigNum::new().unwrap();
        serial.rand(63, MsbOption::ONE, false).unwrap();
        let mut builder = X509Builder::new().unwrap();
        builder.set_version(2).unwrap();
        builder
            .set_serial_number(&serial.to_asn1_integer().unwrap())
            .unwrap();
        builder
            .set_not_before(&Asn1Time::days_from_now(0).unwrap())
            .unwrap();
        builder
            .set_not_after(&Asn1Time::days_from_now(1).unwrap())
            .unwrap();
        builder.set_subject_name(name).unwrap();
        builder.set_issuer_name(issuer).unwrap();
        builder.set_pubkey(key).unwrap();
        builder
    }
}

#[cfg(test)]
mod tests {
    use super::test_pki::{subject, TestCa, TestCert};
    use super::*;
    use openssl::asn1::Asn1Type;
    use tonic::Code;

    fn policy(server_name: &str, organizations: &[&str]) -> Result<PeerCertPolicy, String> {
        let organizations: Vec<String> = organizations.iter().map(|o| o.to_string()).collect();
        PeerCertPolicy::from_config(server_name, &organizations)
    }

    fn check(allowed: &[&str], cert: &TestCert) -> Result<(), String> {
        let allowed: HashSet<String> = allowed.iter().map(|o| o.to_string()).collect();
        check_client_organization(&allowed, &cert.x509())
    }

    #[test]
    fn unset_values_keep_existing_behavior() {
        let policy = policy("", &[]).unwrap();
        assert_eq!(policy, PeerCertPolicy::default());
        assert_eq!(policy.server_name(), DEFAULT_SERVER_NAME);
        for mode in [GrpcTlsMode::Plain, GrpcTlsMode::Mtls, GrpcTlsMode::Psk] {
            assert!(policy.ensure_supported(mode).is_ok(), "{mode:?}");
            assert!(policy.client_organization_interceptor(mode).is_none());
        }
    }

    #[test]
    fn configured_organizations_turn_on_the_check_for_mtls_only() {
        let policy = policy("", &["example-group-a"]).unwrap();
        assert_eq!(policy.server_name(), DEFAULT_SERVER_NAME);
        assert!(policy
            .client_organization_interceptor(GrpcTlsMode::Mtls)
            .is_some());
        for mode in [GrpcTlsMode::Psk, GrpcTlsMode::Plain] {
            assert!(policy.client_organization_interceptor(mode).is_none());
        }
    }

    #[test]
    fn whitespace_only_server_name_is_unset() {
        assert_eq!(policy("  ", &[]).unwrap(), PeerCertPolicy::default());
    }

    #[test]
    fn configured_server_name_is_trimmed_and_used() {
        let policy = policy(" group-a.example.com ", &[]).unwrap();
        assert_eq!(policy.server_name(), "group-a.example.com");
        assert!(policy
            .client_organization_interceptor(GrpcTlsMode::Mtls)
            .is_none());
    }

    #[test]
    fn rejects_server_name_that_is_not_a_dns_name() {
        for name in [
            "10.0.0.1",
            "2001:db8::1",
            "*.example.com",
            "group a.example.com",
            "group-a..example.com",
            "-group-a.example.com",
        ] {
            let err = policy(name, &[]).unwrap_err();
            assert!(err.contains("tls.server_name"), "{name:?}: {err}");
        }
    }

    #[test]
    fn rejects_blank_client_organization() {
        for organizations in [&[""][..], &["example-group-a", "  "][..]] {
            let err = policy("", organizations).unwrap_err();
            let index = organizations.len() - 1;
            assert!(
                err.contains(&format!("tls.allowed_client_organizations[{index}]")),
                "{err}"
            );
        }
    }

    #[test]
    fn checks_require_mtls() {
        for (server_name, organizations, keys) in [
            ("group-a.example.com", &[][..], "tls.server_name"),
            (
                "",
                &["example-group-a"][..],
                "tls.allowed_client_organizations",
            ),
            (
                "group-a.example.com",
                &["example-group-a"][..],
                "tls.server_name and tls.allowed_client_organizations",
            ),
        ] {
            let policy = policy(server_name, organizations).unwrap();
            assert!(policy.ensure_supported(GrpcTlsMode::Mtls).is_ok());
            for mode in [GrpcTlsMode::Psk, GrpcTlsMode::Plain] {
                let err = policy.ensure_supported(mode).unwrap_err();
                assert!(
                    err.starts_with(&format!("{keys} can only be used with mutual TLS")),
                    "{mode:?}: {err}"
                );
            }
        }
    }

    #[test]
    fn interceptor_denies_request_without_client_certificate() {
        let mut interceptor = policy("", &["example-group-a"])
            .unwrap()
            .client_organization_interceptor(GrpcTlsMode::Mtls)
            .unwrap();
        let status = interceptor.call(Request::new(())).unwrap_err();
        assert_eq!(status.code(), Code::PermissionDenied);
        assert_eq!(status.message(), REQUEST_DENIED);
    }

    #[test]
    fn configured_organizations_are_trimmed() {
        let policy = policy("", &[" example-group-a "]).unwrap();
        assert_eq!(policy.allowed_client_organizations, ["example-group-a"]);
        let cert = TestCa::new().client_cert(&["example-group-a"]);
        let allowed: HashSet<String> = policy.allowed_client_organizations.into_iter().collect();
        assert!(check_client_organization(&allowed, &cert.x509()).is_ok());
    }

    #[test]
    fn allows_client_with_allowed_organization() {
        let cert = TestCa::new().client_cert(&["example-group-a"]);
        assert!(check(&["example-group-a"], &cert).is_ok());
    }

    #[test]
    fn allows_client_when_any_organization_is_allowed() {
        let cert = TestCa::new().client_cert(&["example-other", "example-group-a"]);
        assert!(check(&["example-group-b", "example-group-a"], &cert).is_ok());
    }

    #[test]
    fn denies_client_with_other_organization() {
        let cert = TestCa::new().client_cert(&["example-group-b"]);
        let err = check(&["example-group-a"], &cert).unwrap_err();
        assert!(err.contains("example-group-b"), "{err}");
    }

    #[test]
    fn organization_match_is_exact_and_case_sensitive() {
        let ca = TestCa::new();
        for organization in [
            "Example-Group-A",
            "example-group-a ",
            "example-group",
            "example-group-ab",
            "example-group-a\0suffix",
        ] {
            let cert = ca.client_cert(&[organization]);
            assert!(
                check(&["example-group-a"], &cert).is_err(),
                "{organization:?}"
            );
        }
    }

    #[test]
    fn denies_client_without_organization() {
        let cert = TestCa::new().client_cert(&[]);
        let err = check(&["example-group-a"], &cert).unwrap_err();
        assert!(err.contains("no Subject Organization"), "{err}");
    }

    #[test]
    fn reads_organization_stored_as_printable_string() {
        let name = subject(&[(
            Nid::ORGANIZATIONNAME,
            "example-group-a",
            Asn1Type::PRINTABLESTRING,
        )]);
        let cert = TestCa::new().client_cert_with_subject(&name);
        assert!(check(&["example-group-a"], &cert).is_ok());
    }
}

#[cfg(test)]
mod end_to_end_tests {
    //! Serves MNCCD's gRPC service on loopback the way `spawn_and_start_server` does in
    //! mTLS mode, and dials it with tonic's rustls client the way `build_client_channel`
    //! does, so the checks run as they do between nodes.

    use super::test_pki::{TestCa, TestCert};
    use super::{PeerCertPolicy, DEFAULT_SERVER_NAME, REQUEST_DENIED};
    use crate::mnccd_grpc::proto::mnccd_grpc_client::MnccdGrpcClient;
    use crate::mnccd_grpc::proto::mnccd_grpc_server::MnccdGrpcServer;
    use crate::mnccd_grpc::proto::MnccdGrpcEchoRequest;
    use crate::mnccd_grpc::{MnccdGlobalData, MnccdGrpcService};
    use crate::node_allowlist::NodeAllowlistInterceptor;
    use crate::GrpcTlsMode;
    use std::net::SocketAddr;
    use std::time::Duration;
    use tokio::net::TcpListener;
    use tokio::time::timeout;
    use tonic::transport::server::TcpIncoming;
    use tonic::transport::{
        Certificate, Channel, ClientTlsConfig, Endpoint, Identity, Server, ServerTlsConfig,
    };
    use tonic::{Code, Status};

    const LOOPBACK: &str = "127.0.0.1";
    const IO_TIMEOUT: Duration = Duration::from_secs(10);

    fn policy(server_name: &str, organizations: &[&str]) -> PeerCertPolicy {
        let organizations: Vec<String> = organizations.iter().map(|o| o.to_string()).collect();
        PeerCertPolicy::from_config(server_name, &organizations).unwrap()
    }

    /// mTLS server with `policy`'s client check and the `node_ips` allowlist, on an
    /// ephemeral loopback port.
    async fn serve(
        client_ca_pem: &[u8],
        server_cert: &TestCert,
        policy: &PeerCertPolicy,
        node_ips: &[&str],
    ) -> SocketAddr {
        let listener = TcpListener::bind((LOOPBACK, 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        let tls = ServerTlsConfig::new()
            .identity(Identity::from_pem(
                &server_cert.cert_pem,
                &server_cert.key_pem,
            ))
            .client_ca_root(Certificate::from_pem(client_ca_pem));
        let node_ips: Vec<String> = node_ips.iter().map(|ip| ip.to_string()).collect();
        let interceptor = crate::server_interceptor(
            policy.client_organization_interceptor(GrpcTlsMode::Mtls),
            NodeAllowlistInterceptor::from_node_ips(&node_ips).unwrap(),
        );
        let service = MnccdGrpcServer::with_interceptor(
            MnccdGrpcService::new(MnccdGlobalData::new()),
            interceptor,
        );
        let server = Server::builder()
            .tls_config(tls)
            .unwrap()
            .add_service(service)
            .serve_with_incoming(TcpIncoming::from(listener));
        tokio::spawn(server);
        addr
    }

    /// Dials `addr` the way MNCCD dials peers in mTLS mode: the server must present
    /// `server_name` under `ca`, and `client_cert` is this side's identity.
    fn endpoint(
        addr: SocketAddr,
        ca: &TestCa,
        client_cert: &TestCert,
        server_name: &str,
    ) -> Endpoint {
        let tls = ClientTlsConfig::new()
            .domain_name(server_name)
            .ca_certificate(Certificate::from_pem(ca.cert_pem()))
            .identity(Identity::from_pem(
                &client_cert.cert_pem,
                &client_cert.key_pem,
            ));
        Channel::from_shared(format!("https://{addr}"))
            .unwrap()
            .tls_config(tls)
            .unwrap()
    }

    /// Opens a new connection through `endpoint` and sends one echo RPC on it.
    async fn echo(endpoint: &Endpoint) -> Result<(), Status> {
        let channel = timeout(IO_TIMEOUT, endpoint.connect())
            .await
            .expect("connect timed out")
            .map_err(|e| Status::unavailable(format!("connect failed: {e:?}")))?;
        let request = MnccdGrpcEchoRequest {
            name: "test".into(),
        };
        timeout(
            IO_TIMEOUT,
            MnccdGrpcClient::new(channel).mnccd_grpc_echo_message(request),
        )
        .await
        .expect("echo timed out")
        .map(|_| ())
    }

    fn assert_denied(result: Result<(), Status>, context: &str) {
        let status = result.expect_err(context);
        assert_eq!(
            status.code(),
            Code::PermissionDenied,
            "{context}: {status:?}"
        );
        assert_eq!(status.message(), REQUEST_DENIED, "{context}");
    }

    #[tokio::test]
    async fn default_policy_keeps_existing_behavior() {
        let ca = TestCa::new();
        let policy = PeerCertPolicy::default();
        let server_cert = ca.server_cert(DEFAULT_SERVER_NAME);
        let addr = serve(&ca.cert_pem(), &server_cert, &policy, &[LOOPBACK]).await;
        let client_cert = ca.client_cert(&[]);
        let endpoint = endpoint(addr, &ca, &client_cert, policy.server_name());
        echo(&endpoint).await.expect("echo");
    }

    #[tokio::test]
    async fn accepts_peer_that_passes_both_checks() {
        let ca = TestCa::new();
        let policy = policy("group-a.example.com", &["example-group-a"]);
        let server_cert = ca.server_cert("group-a.example.com");
        let addr = serve(&ca.cert_pem(), &server_cert, &policy, &[LOOPBACK]).await;
        let client_cert = ca.client_cert(&["example-group-a"]);
        let endpoint = endpoint(addr, &ca, &client_cert, policy.server_name());
        echo(&endpoint).await.expect("echo");
    }

    #[tokio::test]
    async fn client_rejects_server_without_configured_name() {
        let ca = TestCa::new();
        let server_cert = ca.server_cert("group-b.example.com");
        let addr = serve(
            &ca.cert_pem(),
            &server_cert,
            &PeerCertPolicy::default(),
            &[LOOPBACK],
        )
        .await;
        let client_cert = ca.client_cert(&[]);
        let policy = policy("group-a.example.com", &[]);
        let endpoint_a = endpoint(addr, &ca, &client_cert, policy.server_name());
        let connected = timeout(IO_TIMEOUT, endpoint_a.connect())
            .await
            .expect("connect timed out");
        assert!(
            connected.is_err(),
            "handshake must fail when the server certificate lacks tls.server_name"
        );
        // The same server is reachable under the name its certificate carries.
        let endpoint_b = endpoint(addr, &ca, &client_cert, "group-b.example.com");
        echo(&endpoint_b).await.expect("echo");
    }

    #[tokio::test]
    async fn server_denies_client_without_allowed_organization() {
        let ca = TestCa::new();
        let policy = policy("", &["example-group-a"]);
        let server_cert = ca.server_cert(DEFAULT_SERVER_NAME);
        let addr = serve(&ca.cert_pem(), &server_cert, &policy, &[LOOPBACK]).await;
        for organizations in [&["example-group-b"][..], &[][..]] {
            let client_cert = ca.client_cert(organizations);
            let endpoint = endpoint(addr, &ca, &client_cert, DEFAULT_SERVER_NAME);
            assert_denied(echo(&endpoint).await, &format!("{organizations:?}"));
        }
        // Denied requests don't affect later clients.
        let client_cert = ca.client_cert(&["example-group-a"]);
        let endpoint = endpoint(addr, &ca, &client_cert, DEFAULT_SERVER_NAME);
        echo(&endpoint).await.expect("echo");
    }

    #[tokio::test]
    async fn node_ips_allowlist_still_applies() {
        let ca = TestCa::new();
        let policy = policy("", &["example-group-a"]);
        let server_cert = ca.server_cert(DEFAULT_SERVER_NAME);
        let addr = serve(&ca.cert_pem(), &server_cert, &policy, &["10.0.0.1"]).await;
        let client_cert = ca.client_cert(&["example-group-a"]);
        let endpoint = endpoint(addr, &ca, &client_cert, DEFAULT_SERVER_NAME);
        assert_denied(echo(&endpoint).await, "IP not in node_ips");
    }
}
