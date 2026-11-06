# Webhooks

Webhooks notify your server when events happen in your Acme account, such as a successful charge or a new dispute. Each account can register up to 16 webhook endpoints.

Acme signs every webhook event with the Acme-Signature header so that you can verify that the event came from Acme. Your endpoint must return a 2xx status code within 10 seconds.

If delivery fails, Acme retries the event for up to 3 days with exponential backoff. You can replay any event from the last 30 days in the dashboard.
