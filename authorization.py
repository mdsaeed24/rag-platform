"""Fail-closed checks independent of Qdrant's mandatory retrieval filter."""

from acl_config import DOCUMENT_ACLS


class AuthorizationError(Exception):
    pass


def validate_document_acl(source, acl):
    if not isinstance(acl, dict):
        raise AuthorizationError("Missing document ACL")
    if not isinstance(acl.get("tenant_id"), str) or not acl["tenant_id"].strip():
        raise AuthorizationError("Missing document tenant")
    roles = acl.get("allowed_roles")
    if (
        not isinstance(roles, list) or not roles
        or any(not isinstance(role, str) or role not in {"employee", "hr"} for role in roles)
    ):
        raise AuthorizationError("Invalid document roles")
    if not isinstance(acl.get("classification"), str) or acl["classification"] not in {"internal", "confidential"}:
        raise AuthorizationError("Invalid document classification")
    if acl["classification"] == "confidential" and set(roles) != {"hr"}:
        raise AuthorizationError("Confidential documents must be HR-only")


def authorize_results(results, tenant_id, role):
    """Reject the whole retrieval batch if any chunk fails current policy.

    This supplements pre-retrieval filtering; it never rescues an unfiltered
    query by removing unauthorized hits after retrieval.
    """
    if (
        not isinstance(tenant_id, str) or not tenant_id.strip()
        or not isinstance(role, str) or role not in {"employee", "hr"}
    ):
        raise AuthorizationError("Invalid retrieval identity")
    for result in results:
        payload = result.payload
        if not isinstance(payload, dict):
            raise AuthorizationError("Missing chunk metadata")
        source = payload.get("source")
        acl = DOCUMENT_ACLS.get(source) if isinstance(source, str) else None
        validate_document_acl(source, acl)
        if acl["tenant_id"] != tenant_id or role not in acl["allowed_roles"]:
            raise AuthorizationError("Chunk denied by current document policy")
        # Require the stored metadata to match the current source policy.
        for key in ("tenant_id", "allowed_roles", "classification"):
            if payload.get(key) != acl[key]:
                raise AuthorizationError("Stale or inconsistent chunk ACL")
        if (
            type(payload.get("chunk_id")) is not int or payload["chunk_id"] < 0
            or not isinstance(payload.get("text"), str) or not payload["text"].strip()
        ):
            raise AuthorizationError("Invalid chunk content or identifier")
