All routes should be placed in `routes`. Schemas (inputs) and resources (outputs) that are locally scoped to a route should be placed in the same folder as its route. This ensures that those items are scoped to a certain functionality.

Generic behaviors or dependencies should be placed in `modules`, with the exception that if the behaviors/dependencies are only used in a few (2 - 3) shared places. Dependencies that are specific to API routing or FastAPI internals should be placed in `controllers`.

Basically, we define three layers of abstraction.
`modules -> controllers -> routes`
with routes being the most specific.

This should hopefully help differentiate between dependencies that are local and global.
