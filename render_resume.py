import json
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

#set up directories and paths
BASE_DIR = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "output"

INPUT_JSON = BASE_DIR / "data/new_resume.json"
OUTPUT_HTML = OUTPUT_DIR / "resume.html"

#load generated resume data
def load_resume_data(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

#render html file via template
def render_html(resume_data):
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html","xml"])
    )

    template = env.get_template("resume.html")
    return template.render(resume=resume_data)

#make output directory
OUTPUT_DIR.mkdir(exist_ok=True)

#load resume data and render via html template
resume_data = load_resume_data(INPUT_JSON)
html = render_html(resume_data)

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Resume generated: {OUTPUT_HTML.resolve()}")