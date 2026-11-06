# API rate limits

The Acme API allows up to 100 read requests per second in live mode. Write requests, such as creating charges or refunds, are limited to 50 requests per second in live mode. Test mode has lower limits of 25 requests per second for all operations.

When you exceed a rate limit, the API returns HTTP status 429 Too Many Requests. The response includes a Retry-After header that tells you how many seconds to wait.

Acme recommends exponential backoff with random jitter for retries. Enterprise accounts can ask support for higher limits.
