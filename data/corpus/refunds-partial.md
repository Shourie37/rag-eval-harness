# Partial refunds

Acme supports partial refunds for card and ACH payments. You can issue several partial refunds on one charge until the total refunded amount equals the original charge amount.

To create a partial refund through the API, send a POST request to /v1/refunds with the charge ID and an amount in the smallest currency unit. If you leave out the amount, Acme refunds the full remaining balance of the charge.

Each partial refund creates its own refund object with a unique ID.
