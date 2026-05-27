from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

TESTER_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Passport OCR Tester</title>
    <style>
      :root {
        color-scheme: light;
        font-family: Arial, sans-serif;
        background: #f6f7f9;
        color: #1b1f24;
      }
      body {
        margin: 0;
        padding: 24px;
      }
      main {
        max-width: 1080px;
        margin: 0 auto;
      }
      h1 {
        margin: 0 0 16px;
        font-size: 24px;
      }
      form {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        align-items: center;
        margin-bottom: 18px;
      }
      input[type="file"] {
        max-width: 100%;
      }
      button {
        border: 1px solid #1b1f24;
        background: #1b1f24;
        color: white;
        padding: 9px 14px;
        cursor: pointer;
      }
      button:disabled {
        cursor: wait;
        opacity: 0.65;
      }
      .status {
        min-height: 22px;
        margin-bottom: 14px;
        font-size: 14px;
      }
      .error {
        color: #a40000;
      }
      .media {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 12px;
        margin-bottom: 16px;
      }
      .media-item {
        border: 1px solid #d8dde6;
        background: white;
        padding: 12px;
      }
      .media-item h2 {
        margin: 0 0 8px;
        font-size: 15px;
      }
      .media-item img {
        display: block;
        max-width: 100%;
        max-height: 320px;
        object-fit: contain;
        border: 1px solid #e2e6ee;
        background: #fff;
      }
      pre {
        overflow: auto;
        white-space: pre-wrap;
        word-break: break-word;
        background: #111827;
        color: #f9fafb;
        padding: 14px;
        min-height: 220px;
        font-size: 13px;
      }
    </style>
  </head>
  <body>
    <main>
      <h1>Passport OCR Tester</h1>
      <form id="upload-form">
        <input id="passport-file" name="file" type="file" accept="image/*,.pdf" required>
        <button id="submit-button" type="submit">Upload</button>
      </form>
      <div id="status" class="status"></div>
      <section id="media" class="media"></section>
      <pre id="output">{}</pre>
    </main>
    <script>
      const form = document.getElementById("upload-form");
      const fileInput = document.getElementById("passport-file");
      const submitButton = document.getElementById("submit-button");
      const statusBox = document.getElementById("status");
      const mediaBox = document.getElementById("media");
      const outputBox = document.getElementById("output");

      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const file = fileInput.files[0];
        if (!file) {
          statusBox.textContent = "Choose a passport image or PDF first.";
          statusBox.className = "status error";
          return;
        }

        const formData = new FormData();
        formData.append("file", file);
        submitButton.disabled = true;
        statusBox.textContent = "Uploading and extracting...";
        statusBox.className = "status";
        mediaBox.innerHTML = "";
        outputBox.textContent = "{}";

        try {
          const response = await fetch("/v1/passports/ocr", {
            method: "POST",
            body: formData,
          });
          const payload = await response.json();
          if (!response.ok) {
            throw new Error(JSON.stringify(payload, null, 2));
          }
          renderMedia(payload.images || {});
          outputBox.textContent = JSON.stringify(scrubLargeImageData(payload), null, 2);
          statusBox.textContent = "Done";
        } catch (error) {
          statusBox.textContent = "Request failed";
          statusBox.className = "status error";
          outputBox.textContent = error instanceof Error ? error.message : String(error);
        } finally {
          submitButton.disabled = false;
        }
      });

      function renderMedia(images) {
        for (const name of ["portrait", "signature"]) {
          const item = images[name];
          if (!item || !item.present || !item.data_base64) {
            continue;
          }
          const article = document.createElement("article");
          article.className = "media-item";
          const title = document.createElement("h2");
          title.textContent = `${name} (${item.method}, confidence ${item.confidence})`;
          const image = document.createElement("img");
          image.alt = name;
          image.src = `data:${item.content_type};base64,${item.data_base64}`;
          article.append(title, image);
          mediaBox.append(article);
        }
      }

      function scrubLargeImageData(payload) {
        return JSON.parse(JSON.stringify(payload, (key, value) => {
          if (key === "data_base64" && typeof value === "string") {
            return `[base64 omitted from text view, ${value.length} chars]`;
          }
          return value;
        }));
      }
    </script>
  </body>
</html>
"""


@router.get("/tester", response_class=HTMLResponse)
async def manual_tester() -> str:
    return TESTER_HTML
