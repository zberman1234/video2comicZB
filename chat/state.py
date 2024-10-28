import os
import reflex as rx
from openai import OpenAI
from chat.video2pdf import get_pdf, get_PIL_from_pdf


class State(rx.State):
    
    output_path: str = 'comic_output.pdf'
    video_path: str
    OPENAI_API: str
    GEMINI_API: str
    STYLE_API: str
    pdf_images = []

    def update_openai(self, new_api):
        self.OPENAI_API = new_api
    def update_gemini(self, new_api):
        self.GEMINI_API = new_api
    def update_style(self, new_api):
        self.STYLE_API = new_api



    async def handle_upload(
        self, files: list[rx.UploadFile]
        ):
        """Handle the upload of file(s).

        Args:
            files: The uploaded files.
        """
        for file in files:
            upload_data = await file.read()
            outfile = rx.get_upload_dir() / file.filename

            print("uploaded file")
            print(outfile)

            self.video_path = outfile

            # Save the file.
            with outfile.open("wb") as file_object:
                file_object.write(upload_data)

    def get_pdf_(self, ):    
        get_pdf(self.video_path, self.OPENAI_API, self.GEMINI_API, self.STYLE_API, self.output_path)
        self.pdf_images = get_PIL_from_pdf(self.output_path)
        os.remove(self.video_path)