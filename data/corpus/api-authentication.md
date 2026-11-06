# API authentication

The Acme API uses API keys to authenticate requests. Secret keys start with sk_live_ in live mode and sk_test_ in test mode. Publishable keys start with pk_ and are safe to use in browser code.

Never share a secret key or commit it to source control. You can roll a secret key from the dashboard, and the old key stays valid for 12 hours so you can deploy the new key.

Restricted keys let you limit a key to specific resources and permissions. Send the key in the Authorization header as a Bearer token.
