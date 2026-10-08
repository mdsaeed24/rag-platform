DOCUMENT_ACLS = {
    "data/acme/handbook.txt": {
        "tenant_id": "acme",
        "allowed_roles": ["employee", "hr"],
        "classification": "internal",
    },

    "data/acme/salaries.txt": {
        "tenant_id": "acme",
        "allowed_roles": ["hr"],
        "classification": "confidential",
    },

    "data/globex/handbook.txt": {
        "tenant_id": "globex",
        "allowed_roles": ["employee", "hr"],
        "classification": "internal",
    },

    "data/globex/salaries.txt": {
        "tenant_id": "globex",
        "allowed_roles": ["hr"],
        "classification": "confidential",
    },
}