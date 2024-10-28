import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from openai import OpenAI
import time
import json
import google.generativeai as genai
import os
import io
import ast
import base64
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
from ultralytics import YOLO
import requests
from fpdf import FPDF
import fitz  # PyMuPDF
from io import BytesIO

def get_PIL_from_pdf(pdf_path):
  pdf_document = fitz.open(pdf_path)

  pil_images = []
  for page_number in range(len(pdf_document)):
    # Get the page
    page = pdf_document[page_number]

    # Render the page to a pixmap
    pix = page.get_pixmap()

    # Convert the pixmap to a bytes object
    img_bytes = pix.tobytes("png")

    # Convert the bytes object to a PIL Image
    pil_image = Image.open(BytesIO(img_bytes))
    pil_images.append(pil_image)
  return pil_images

def bytes_from_im(img):
  is_success, buffer = cv2.imencode(".jpg", img)
  io_buf = io.BytesIO(buffer)
  # decode
  decode_img = cv2.imdecode(np.frombuffer(io_buf.getbuffer(), np.uint8), -1)
  return base64.b64encode(buffer).decode("utf-8")

def get_request(prompt, images):
  buffers = [bytes_from_im(img) for img in images]
  btes_images = [bytes_from_im(img) for img in images]
  image_requests = [[{
            "type": "text",
            "text": prompt + f'{i}'
          },{
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{btes}"
            }}] for i, btes in enumerate(btes_images)]
  return image_requests

def get_timestamps(model, video_path):
  file = genai.upload_file(video_path)
  # Videos need to be processed before you can use them.
  while file.state.name == "PROCESSING":
      print("processing video...")
      time.sleep(5)
      file = genai.get_file(file.name)

  prompt = '''I’m trying to make a comic strip that summarizes this video. Give me key timestamps in MM:SS of the video from which I can draw the comic panels.\
  The JSON format you should output is as follows: a list  of [ { \"timestamp\": \"MM:SS\", \"reason\": reason_for_timestamp,\
    \"panel_dialogue\": string dialogue that is relevant to the panel at that timestamp} ]

    Only output the json string without any other text. Make the output parseable in Python'''
  # "Explain this video clip in a series of comic strip panel descriptions. Include important quotes in the descriptions. The quotes must contain less than 5 words. Also include important visual details. In a JSON format--a list of {\"panel\": panel_number, \"description\": description_with_key_dialogue}."
  result = model.generate_content([file, prompt])
  #print(result.text)
  try:
    final_json = json.loads(result.text[8:-4])
  except:
    final_json = json.loads(result.text)
  return final_json

def get_frame_at_timestamp(video_path, timestamp):
  # Load the video
  cap = cv2.VideoCapture(video_path)

  # Check if the video was loaded successfully
  if not cap.isOpened():
      print("Error: Could not open the video.")
      exit()


  # Calculate the frame number based on the video's frames per second (fps)
  fps = cap.get(cv2.CAP_PROP_FPS)
  frame_number = int(fps * timestamp)

  frames = []

  for i in range(-3, 3, 1):
    # Set the video to the calculated frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number + i * int(fps/4)))

    # Read the frame
    ret, frame = cap.read()
    frames.append(frame)

  # Release the video capture
  cap.release()
  return frames

def mmss_to_seconds(mmss):
    minutes, seconds = map(int, mmss.split(':'))
    return minutes * 60 + seconds

def identify_keyframe(client, prompt, request):
  out = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "system", "content": prompt}] + [{'role':'user', 'content':im_req} for im_req in request],
    temperature=0.000001
  )
  topics = out.choices[0].message.content
  return topics

def get_video_frames(client, video_path, text_json, offset=0.5):
  final_frames = []
  prompt_keyframe = """I will give you a number of frames from a small segment of a video. I need you to identify the best-looking frame out of all of these \
so I can use it as a basis for a comic book panel created from the video. The frame needs to have no blur. Only output the number of the best frame, as an integer without any additional words."""
  for panel in text_json:
    timestamp = mmss_to_seconds(panel['timestamp'])
    description = panel['panel_dialogue']
    frames = get_frame_at_timestamp(video_path, timestamp+offset)
    req = get_request(f"Here is one of the frames you need to analyze. For some context, this is the dialogue that is attached to the picture: {description}. This is frame number ", frames)
    frame_id = int(identify_keyframe(client, prompt_keyframe, req))
    frame = frames[frame_id]
    final_frames.append((frame[...,::-1], description))
  return final_frames


def get_boxes(model, image):
  results = model(image)[0].boxes
  people = []
  for i, cls in enumerate(results.cls):
    if results.cls[i] == 0:
      people.append(list(np.array(results.xyxy[i])))
  return np.array(people)

def get_center_person(model, image, frame):
  results = get_boxes(model, image)
  if len(results) == 0:
    return None
  centers = (results[:, :2] + results[:, 2:] )/ 2
  centers = centers - (np.array(frame.shape)[:2] / 2)
  centers_dist = np.linalg.norm(centers, axis=1)
  centered = np.argmin(centers_dist)
  box = results[centered]
  return box


def draw_text_bubble(model_yolo, frame, img, desc):
  center_box = get_center_person(model_yolo, frame, frame)

  factor_x = frame.shape[1] / img.size[0]
  factor_y = frame.shape[0] / img.size[1]

  img_width = img.size[0]
  max_width = img_width * 0.5
  font_size = int(max_width/9.5)
  font = ImageFont.load_default(size=font_size)

  if (not (center_box is None)) and (desc is not None):
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = center_box
    y_top = min(y1, y2) / factor_y
    x_mid = (x1+x2)/2
    x_mid /= factor_x

    # Get text sizes
    lines = [""]
    line_lengths = []
    line_length=0
    for word in desc.split():
      lines[-1] += (" " + word)
      line_length = draw.textlength(lines[-1], font=font)
      if line_length > max_width:
        line_lengths.append(line_length)
        lines.append("")
    line_lengths.append(line_length)

    width = max(line_lengths)
    height = font_size * len(lines)

    # Get position/size of the bounding speech bubble
    text_box = [x_mid, y_top - height, x_mid + width, y_top]

    right_overflow = text_box[2] - frame.shape[1] / factor_x
    if right_overflow > 0:
      text_box[0] -= (right_overflow + 50)
      text_box[2] -= (right_overflow + 50)

    top_overflow = 0 - text_box[3]
    if top_overflow > 0:
      text_box[1] += top_overflow*1.2
      text_box[3] += top_overflow*1.2

    # draw speech bubble
    draw.rounded_rectangle([text_box[0] - 10, text_box[1], text_box[0] - 10 + width*1.1, text_box[1]+height*1.1], fill="white", outline="black", radius=15, width=5)

    # draw text inside speech bubble
    for i, line in enumerate(lines):
      draw.text((text_box[0], text_box[1] + i*font_size*1.05), line, font=font, fill="black")
  return img


def cartoonify(frame, STYLE_API):
    MAGIC_API_KEY = STYLE_API
    cv2.imwrite("frame.png", frame)
    url = "https://api.magicapi.dev/api/v1/ailabtools/ai-cartoon-generator/image/effects/generate_cartoonized_image"
    headers = {"x-magicapi-key": MAGIC_API_KEY}
    files = {
        "task_type": (None, "async"),
        "image": ("frame.png", open("frame.png", "rb"), "image/png"),
        "index": (None, "0"),  # style index (see docs)
    }

    response = requests.post(url, headers=headers, files=files)
    response_dict = response.json()
    task_id = response_dict.get("task_id")
    print(response.text)

    url = f"https://api.magicapi.dev/api/v1/ailabtools/ai-cartoon-generator/api/apimarket/query-async-task-result?task_id={task_id}"
    headers = {"x-magicapi-key": MAGIC_API_KEY}
    task_status = 1
    while task_status != 2:  # wait until status = 2 (completed)
        response = requests.get(url, headers=headers)
        response_dict = response.json()
        print(response_dict)
        task_status = response_dict.get("task_status", 0)
        # print(response.text)
        if task_status != 2:
            print(task_status)
            time.sleep(1)

    print("Processing completed, handling image...")
    image_url = response_dict.get("data", {}).get("result_url")

    # download
    image_response = requests.get(image_url)
    img_data = np.frombuffer(image_response.content, np.uint8)
    return cv2.imdecode(img_data, cv2.IMREAD_COLOR)[..., ::-1]


def add_boundaries(img):
    # Original image dimensions
    original_width, original_height = img.size

    # Calculate 5% black boundary size
    black_border_size = int(0.03 * original_width), int(0.03 * original_height)

    # Add black border
    img_with_black_border = ImageOps.expand(
        img,
        border=black_border_size,
        fill='black'
    )

    # Calculate new dimensions after black border
    new_width, new_height = img_with_black_border.size

    # Calculate 10% white boundary based on new dimensions
    white_border_size = int(0.06 * new_width), int(0.06 * new_height)

    # Add white border
    final_img = ImageOps.expand(
        img_with_black_border,
        border=white_border_size,
        fill='white'
    )

    return final_img


def resize_image(img, max_width, target_height):
    """
    Resize the image to have the specified target height while maintaining aspect ratio.
    If the resulting width exceeds max_width, resize again to fit within max_width.
    """
    width, height = img.size

    # Calculate new dimensions maintaining aspect ratio
    aspect_ratio = width / height
    new_width = int(target_height * aspect_ratio)
    new_height = target_height

    # Resize if necessary
    if new_width > max_width:
        new_width = max_width
        new_height = int(max_width / aspect_ratio)

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img

def create_comic_pdf(images, output_pdf):
    # PDF dimensions
    pdf_width, pdf_height = 816, 1056
    image_height = int(0.45 * pdf_height)  # 40% of the PDF height

    # Initialize PDF
    pdf = FPDF(unit="pt", format=(pdf_width, pdf_height))
    pdf.set_auto_page_break(auto=True, margin=0)

    # Track position on the page
    x, y = 0, 0

    for img_ in images:
        # Resize the image to fit the constraints
        img = resize_image(img_, max_width=pdf_width, target_height=image_height)



        # Update the x, y position for the next image
        if x + img.width > pdf_width:
            x = 0
            y += image_height

        # Check if we need a new page
        if y + image_height > pdf_height:
            # Reset position for new page
            x, y = 0, 0
        # Add new page if we're at the top left
        if x == 0 and y == 0:
            pdf.add_page()
        # Add image to PDF
        pdf.image(img, x=x, y=y)

        x += img.width

    # Save the final PDF
    pdf.output(output_pdf)

def get_pdf(video_path, OPENAI_API, GEMINI_API, STYLE_API, output_path="comic_output.pdf"):
  model_yolo = YOLO("yolov9c.pt")
  model = genai.GenerativeModel("gemini-1.5-pro")

  client = OpenAI(api_key=OPENAI_API)
  genai.configure(api_key=GEMINI_API)

  text_json = get_timestamps(model, video_path)

  frames = get_video_frames(client, video_path, text_json, offset=1)

  final_imgs = []
  for (frame, desc) in frames:
    img = Image.fromarray(cartoonify(frame, STYLE_API))
    img = draw_text_bubble(model_yolo, frame, img, desc)

    final_imgs.append(img)

  for i, img in enumerate(final_imgs):
    boundary_img = add_boundaries(img)
    final_imgs[i] = boundary_img

  create_comic_pdf(final_imgs, output_path)