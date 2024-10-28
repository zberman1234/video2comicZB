import reflex as rx

from chat.components import loading_icon
from chat.state import State

def chat() -> rx.Component:
    """List all the messages in a single conversation."""
    return rx.vstack(
        rx.foreach(
            State.pdf_images, rx.image
        )
    )


def action_bar() -> rx.Component:
    """The action bar to send a new message."""
    return rx.hstack(
        rx.upload(
            rx.vstack(
                rx.button(
                    "Select Video",
                    color="rgb(107,99,246)",
                    bg="white",
                    border="1px solid rgb(107,99,246)",
                )
            ),
            id="upload1",
            multiple=False,
            accept={
                "video/mp4": [".mp4"],
                "video/quicktime": [".mov"],
            },
            max_files=1,
            padding="0px",
        ),
        rx.hstack(
            rx.foreach(
                rx.selected_files("upload1"), rx.text
            )
        ),
        rx.button(
            "Upload",
            on_click=State.handle_upload(
                rx.upload_files(upload_id="upload1")
            ),
        ),
        rx.button(
            "Process Video",
            on_click=State.get_pdf_()
        ),
        rx.button(
            "Download Comic",
            on_click=rx.download(
                url="/",
                filename=State.output_path,
            ),
        ),
        padding="5em",
    )
