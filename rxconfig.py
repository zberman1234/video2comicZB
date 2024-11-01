import os
import reflex as rx

config = rx.Config(
    app_name="chat",
    api_url="/api",
    backend_port=int(os.environ.get("PORT", 8000)),
    frontend_port=int(os.environ.get("PORT", 3000)),
)
