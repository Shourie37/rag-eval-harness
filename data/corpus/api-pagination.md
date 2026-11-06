# Pagination

List endpoints in the Acme API return results in pages. Use the limit parameter to set the page size. The default limit is 10 objects, and the maximum limit is 100 objects.

To get the next page, pass the ID of the last object in the starting_after parameter. Each list response includes a has_more field that is true when more results exist.

Results are sorted by creation time, with the newest objects first.
