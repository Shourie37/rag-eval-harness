# API errors

The Acme API uses standard HTTP status codes to show whether a request succeeded.

- A 400 status means the request was invalid, often because a required parameter is missing.
- A 401 status means the API key is missing or invalid.
- A 402 status means the request was valid but the payment failed, such as a declined card.
- A 404 status means the requested object does not exist.
- A 500 status means something went wrong on the Acme side, and these errors are rare.

Every error response includes a type, a code, and a human-readable message.
