# Idempotent requests

The Acme API supports idempotency so that you can retry requests safely. To make a request idempotent, send an Idempotency-Key header with a unique value such as a UUID.

Acme stores the result of the first request for each key for 24 hours. If you send the same key again within that time, Acme returns the stored result instead of creating a duplicate charge.

Idempotency keys can be up to 255 characters long. GET and DELETE requests are idempotent by design and do not need a key.
