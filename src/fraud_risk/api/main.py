"""ASGI application factory.

Responsibility
    Create the FastAPI app, wire the lifespan (load the model once, dispose the engine on
    shutdown), install structured logging with request ids and register the routers.

Routes (milestone M6)
    ``GET  /health``            process health and model availability, reported separately
    ``POST /predict``           score one transaction and persist the result
    ``GET  /predictions/{id}``  read a stored prediction back
    ``POST /feedback``          attach a confirmed outcome to a stored prediction

Status
    Scaffolding only. See ``docs/ROADMAP.md`` (M6).
"""
