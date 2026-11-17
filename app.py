"""MedExtract Streamlit interface. Run: streamlit run app.py"""

import html
import json
import logging
import re
import tempfile
from collections import Counter
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src.extractor import extract_entities
from src.input_handler import InputProcessingError, load_input


MODEL_PATH = Path(__file__).resolve().parent / "models" / "medical_ner.pkl"


@st.cache_resource(show_spinner=False)
def ensure_model() -> str:
    """Build a missing model once per app process, using training data only.

    Streamlit serializes concurrent calls for this cache entry. Existing models
    are left untouched, and paths do not depend on the server's working folder.
    """
    if not MODEL_PATH.is_file():
        import train

        train.main()
        if not MODEL_PATH.is_file():
            raise RuntimeError("Training completed without creating the model artifact.")
    return str(MODEL_PATH)


ENTITY_COLORS = {
    "MEDICINE": "#BAE6FD", "TEST": "#A7F3D0",
    "SYMPTOM": "#FED7AA", "DATE": "#E0F2FE",
}
ENTITY_NAMES = {"MEDICINE": "Medicines", "TEST": "Tests", "SYMPTOM": "Symptoms", "DATE": "Dates"}
OCEAN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=DM+Mono:wght@400;500&display=swap');
:root { --ocean-bg:#F0F7F9; --ocean-surface:#FFFFFF; --ocean-primary:#0EA5B0;
 --ocean-hover:#38BEC9; --ocean-text:#0D2B35; --ocean-muted:#4A7A85; --ocean-border:#C8E6EA; }
html,body,.stApp,.stApp h1,.stApp h2,.stApp h3,.stApp p,.stApp label,.stApp input,.stApp textarea,.stApp button,.stApp [data-testid="stMarkdownContainer"] { font-family:'DM Sans',sans-serif; }
[data-testid="stAppViewContainer"], .stApp { background:var(--ocean-bg); color:var(--ocean-text); }
[data-testid="stHeader"] { background:var(--ocean-bg); }
.block-container { max-width:1120px; padding:2.5rem 2rem 3rem; }
h1,h2,h3,p,label { color:var(--ocean-text); }
[data-testid="stCaptionContainer"] p { color:var(--ocean-muted); }
.brand { display:flex; align-items:center; gap:12px; height:48px; font-size:25px; font-weight:750; letter-spacing:-.7px; }
.brand-icon { width:42px; height:42px; border-radius:13px; background:var(--ocean-primary); display:grid; place-items:center; }
.brand-subtitle { font-size:10px; letter-spacing:1.8px; font-weight:650; color:var(--ocean-muted); margin-top:-3px; }
.hero { padding:34px 0 23px; }
.eyebrow { color:var(--ocean-primary); font-size:11px; font-weight:750; letter-spacing:2px; text-transform:uppercase; }
.hero h1 { font-size:clamp(28px,4vw,40px); letter-spacing:-1.4px; line-height:1.2; font-weight:650; margin:9px 0 12px; padding:0; }
.hero p { color:var(--ocean-muted); font-size:15px; max-width:650px; line-height:1.7; margin:0; }
.st-key-input_card { background:var(--ocean-surface); border:1px solid var(--ocean-border); border-radius:22px; padding:25px; box-shadow:0 8px 32px rgba(13,43,53,.035); }
.card-heading { font-size:18px; font-weight:650; letter-spacing:-.4px; margin-bottom:3px; }
.card-description { color:var(--ocean-muted); font-size:13px; margin-bottom:16px; }
[data-testid="stTextArea"] textarea { background:var(--ocean-surface); color:var(--ocean-text); line-height:1.75; font-size:15px; border:1px solid var(--ocean-border); border-radius:12px; padding:14px; outline:none; transition:border-color 0.2s ease,box-shadow 0.2s ease; }
[data-testid="stTextArea"] textarea:focus { border-color:var(--ocean-primary); box-shadow:0 0 0 3px rgba(14,165,176,.10); }
[data-testid="stTextArea"] > div:has(> textarea) { border-color:var(--ocean-border); border-radius:12px; transition:border-color 0.2s ease; }
[data-testid="stTextArea"] > div:has(> textarea):focus-within { border-color:var(--ocean-primary); }
[data-testid="stTextArea"] [data-baseweb="textarea"] { border:1px solid var(--ocean-border); border-radius:12px; background:var(--ocean-surface); transition:border-color 0.2s ease,box-shadow 0.2s ease; }
[data-testid="stTextArea"] [data-baseweb="textarea"]:focus-within { border-color:var(--ocean-primary); box-shadow:0 0 0 3px rgba(14,165,176,.10); }
[data-testid="stFileUploader"] { display:flex; flex-direction:column; align-items:center; text-align:center; width:100%; }
[data-testid="stFileUploaderDropzone"] { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:12px; width:100%; min-height:260px; background:var(--ocean-bg); border:2px dashed var(--ocean-border); border-radius:14px; padding:28px 20px; text-align:center; transition:border-color 200ms ease,background-color 200ms ease; }
[data-testid="stFileUploaderDropzone"]:hover,[data-testid="stFileUploaderDropzone"]:focus-within { border-color:var(--ocean-primary); background:var(--ocean-surface); }
[data-testid="stFileUploaderDropzone"]::before { content:"Drag & drop your file here"; order:1; display:block; padding-top:50px; background:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='34' height='34' viewBox='0 0 24 24' fill='none' stroke='%230EA5B0' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 16V3m-5 5 5-5 5 5M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4'/%3E%3C/svg%3E") center top / 34px 34px no-repeat; color:var(--ocean-text); font-family:'DM Sans',sans-serif; font-size:15px; font-weight:500; pointer-events:none; }
[data-testid="stFileUploaderDropzone"]::after { content:"or"; order:2; color:var(--ocean-muted); font-family:'DM Sans',sans-serif; font-size:13px; pointer-events:none; }
[data-testid="stFileUploaderDropzone"] > span:has(button),[data-testid="stFileUploaderDropzone"] > button { order:3; align-self:center; margin:0; }
[data-testid="stFileUploaderDropzone"] > span:has(button) { display:flex; justify-content:center; }
[data-testid="stFileUploaderDropzoneInstructions"] { order:4; display:flex; flex-direction:column; align-items:center; text-align:center; }
[data-testid="stFileUploaderDropzoneInstructions"] > div { display:flex; justify-content:center; align-items:center; }
[data-testid="stFileUploaderDropzoneInstructions"] span { font-family:'DM Mono',monospace; font-size:11px; color:var(--ocean-muted); }
.st-key-input_card [data-testid="stCaptionContainer"] { text-align:center; }
[data-testid="stFileUploader"] small,[data-testid="stFileUploaderFileName"] { font-family:'DM Mono',monospace; font-size:11px; font-weight:400; color:var(--ocean-muted); }
[data-testid="stRadio"] [role="radiogroup"] { gap:24px; }
[data-testid="stRadio"] label:has(input:checked) { color:var(--ocean-primary); }
[data-testid="stRadio"] input { accent-color:var(--ocean-primary); }
[data-testid="stRadio"] label:has(input:checked) > div:first-child { background-color:var(--ocean-primary); }
[data-testid="stRadioOption"] > div > div:first-child { background:var(--ocean-surface); border:1px solid var(--ocean-border); }
[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child { background:var(--ocean-primary); border-color:var(--ocean-primary); }
[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child > div { background:var(--ocean-surface); }
button[kind="primary"] { background:var(--ocean-primary); border:1px solid var(--ocean-primary); color:var(--ocean-surface); border-radius:11px; min-height:46px; font-weight:600; box-shadow:0 4px 12px rgba(14,165,176,.14); transition:background 180ms ease-out,transform 200ms ease-out,box-shadow 200ms ease-out; }
button[kind="primary"] p { color:var(--ocean-surface); }
button[kind="primary"]:not(:disabled):hover { background:var(--ocean-hover); border-color:var(--ocean-hover); transform:scale(1.01); box-shadow:0 0 0 5px rgba(14,165,176,.10),0 6px 20px rgba(14,165,176,.20); }
.st-key-extract_busy button { opacity:1; cursor:wait; gap:10px; }
.st-key-extract_busy button::before { content:""; display:inline-block; width:15px; height:15px; flex-shrink:0; border:2px solid rgba(255,255,255,.35); border-top-color:#FFFFFF; border-radius:50%; animation:button-spin 300ms linear infinite; }
button[kind="secondary"] { background:var(--ocean-surface); border:1px solid var(--ocean-border); color:var(--ocean-text); border-radius:10px; transition:background 180ms,border-color 180ms; }
button[kind="secondary"]:hover { border-color:var(--ocean-hover); background:var(--ocean-bg); color:var(--ocean-text); }
button:focus-visible { outline:2px solid var(--ocean-primary)!important; outline-offset:3px; }
[data-testid="stAlert"] { background:var(--ocean-surface); border:1px solid var(--ocean-border); border-radius:12px; color:var(--ocean-text); }
.ready-state { text-align:center; border:1px dashed var(--ocean-border); border-radius:18px; padding:27px; margin-top:22px; }
.ready-state p { color:var(--ocean-muted); font-size:13px; margin:6px 0 0; }
.footer-note { font-size:11px; color:var(--ocean-muted); margin:22px 0 0; text-align:center; }
@keyframes input-rise { from { transform:translateY(12px); opacity:0; } to { transform:translateY(0); opacity:1; } }
@keyframes button-spin { to { transform:rotate(360deg); } }
@media(max-width:640px) { .block-container { padding:1.5rem 1rem; } .st-key-input_card { padding:18px; } .hero { padding-top:24px; } }
@media(prefers-reduced-motion:reduce) { *,*::before,*::after { animation:none!important; transition:none!important; } }
</style>
"""


def clear_results() -> None:
    """Invalidate results when the user changes the document or source."""
    st.session_state.pop("result", None)
    st.session_state.pop("notice", None)


def read_uploaded_file(uploaded_file) -> str:
    """Adapt Streamlit uploads to the existing path-based input handler."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {".txt", ".pdf"}:
        raise InputProcessingError("Please choose a TXT or PDF file.")
    # Use a generated name, never an uploaded filename as a filesystem path.
    with tempfile.TemporaryDirectory(prefix="medextract-") as directory:
        path = Path(directory) / ("document" + suffix)
        path.write_bytes(uploaded_file.getvalue())
        return load_input(path)


def display_text(text: str) -> str:
    """Escape document text and style numeric values without changing its content."""
    parts = []
    cursor = 0
    for match in re.finditer(r"\d+(?:[.,:/-]\d+)*", text):
        parts.append(html.escape(text[cursor:match.start()]))
        parts.append(f'<span class="data-value">{html.escape(match[0])}</span>')
        cursor = match.end()
    parts.append(html.escape(text[cursor:]))
    return "".join(parts)


def results_html(text: str, entities: list[dict]) -> str:
    """Build an isolated, escaped results view with linked spans and table rows."""
    counts = Counter(entity["label"] for entity in entities)
    pills = "".join(
        f'<div class="pill"><span class="dot" style="background:{color}"></span>'
        f'<span>{ENTITY_NAMES[label]}</span><strong data-count="{counts[label]}">{counts[label]}</strong></div>'
        for label, color in ENTITY_COLORS.items()
    )
    parts, rows = [], []
    cursor = 0
    for index, entity in enumerate(entities):
        start, end, label = entity["start"], entity["end"], entity["label"]
        color = ENTITY_COLORS[label]
        parts.append(display_text(text[cursor:start]))
        parts.append(
            f'<mark id="entity-{index}" class="entity" tabindex="0" data-id="{index}" '
            f'title="{label}" aria-label="{html.escape(entity["text"], quote=True)}: {label}" '
            f'style="--badge:{color};--delay:{index * 40}ms">'
            f'{display_text(text[start:end])}</mark>'
        )
        cursor = end
        rows.append(
            f'<tr tabindex="0" data-id="{index}" style="--delay:{index * 40}ms" aria-label="{html.escape(entity["text"], quote=True)}, {label}">'
            f'<td class="entity-name">{display_text(entity["text"])}</td>'
            f'<td><span class="badge" style="background:{color}">{label}</span></td>'
            f'<td class="number">{start}</td><td class="number">{end}</td>'
            f'<td><span class="method">{html.escape(entity["method"])}</span></td></tr>'
        )
    parts.append(display_text(text[cursor:]))
    empty = '<p class="empty">No entities found in this document. Try a note containing a medicine, test, symptom, or date.</p>' if not entities else ""
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=DM+Mono:wght@400;500&display=swap');
*{box-sizing:border-box} body{margin:0;padding:7px 2px 12px;background:#F0F7F9;color:#0D2B35;font-family:'DM Sans',sans-serif;font-size:14px}
.results{animation:panel-in 300ms ease-out} h2{font-size:22px;font-weight:650;letter-spacing:-.6px;margin:0} .heading{display:flex;align-items:center;justify-content:space-between;gap:14px;margin:17px 0}
.muted{color:#4A7A85;font-size:12px} .summary{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0 24px}
.pill{background:#FFFFFF;border:1px solid #C8E6EA;border-radius:14px;padding:16px;display:flex;align-items:center;gap:9px;font-size:12px;color:#4A7A85}
.pill strong{font-size:12px;font-weight:500;color:#0D2B35;margin-left:auto;font-variant-numeric:tabular-nums}.dot{height:10px;width:10px;border-radius:50%;flex-shrink:0}
.card{background:#FFFFFF;border:1px solid #C8E6EA;border-radius:18px;box-shadow:0 6px 24px rgba(13,43,53,.025);overflow:hidden;margin-bottom:18px}
.card-head{padding:19px 23px 13px;display:flex;justify-content:space-between;align-items:center;gap:12px}.card-head h3{font-size:15px;letter-spacing:-.2px;font-weight:650;margin:0}
.annotated{white-space:pre-wrap;overflow-wrap:anywhere;line-height:2.05;padding:4px 23px 22px;max-height:255px;overflow-y:auto;font-size:15px}
mark{background-color:transparent;color:#0D2B35;border-radius:4px;padding:2px 1px;box-decoration-break:clone;-webkit-box-decoration-break:clone;transition:background-color 250ms ease-out var(--delay),box-shadow 180ms ease-out,filter 180ms ease-out;cursor:default}
body.revealed mark{background-color:var(--badge)}
mark.linked,mark:focus-visible{box-shadow:0 0 0 2px #0EA5B0,0 0 0 5px rgba(14,165,176,.12);outline:none;filter:brightness(.98)}
.table-wrap{max-height:305px;overflow:auto}table{border-collapse:collapse;width:100%;text-align:left;font-size:13px}th{position:sticky;top:0;background:#F0F7F9;color:#4A7A85;font-size:10px;letter-spacing:1px;text-transform:uppercase;font-weight:600;z-index:1}
td,th{padding:12px 23px;border-top:1px solid #C8E6EA}tbody tr{animation:row-in 240ms ease-out both;animation-delay:var(--delay);transition:background 180ms ease-out;outline:none}tbody tr:hover,tbody tr.linked,tbody tr:focus-visible{background:#F0F7F9}tbody tr:focus-visible{box-shadow:inset 3px 0 #0EA5B0}
.entity-name{font-weight:550;overflow-wrap:anywhere;max-width:380px}.number{color:#4A7A85;font-variant-numeric:tabular-nums}.badge{font-size:11px;letter-spacing:.3px;font-weight:500;border-radius:6px;padding:5px 8px;white-space:nowrap}.method{font-size:11px;color:#4A7A85}
.metadata,.number,.method,.badge,.data-value,.pill strong,.pill > span:not(.dot),th{font-family:'DM Mono',monospace;font-size:12px;font-weight:400;font-variant-numeric:tabular-nums}
.badge,.method,th{font-size:11px}.badge,.pill strong{font-weight:500}
.empty{color:#4A7A85;line-height:1.7;padding:0 23px 16px}.hint{color:#4A7A85;font-size:11px;margin:0 2px}
@keyframes panel-in{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes row-in{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
@media(max-width:650px){.summary{grid-template-columns:repeat(2,1fr);gap:8px}.pill{padding:12px}.card-head{align-items:flex-start;flex-direction:column}td,th{padding:12px}.annotated{padding:6px 16px 20px}.heading{align-items:flex-start;flex-direction:column}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important}mark{background-color:var(--badge)}}
</style></head><body><section class="results" aria-label="Extraction results">
""" + (
        f'<div class="heading"><h2>Extraction results</h2><span class="muted metadata">{len(entities)} entities · {len(text):,} characters</span></div>'
        f'<div class="summary">{pills}</div>'
        '<div class="card"><div class="card-head"><h3>Annotated document</h3><span class="muted">Your text, with entities in context</span></div>'
        f'<div class="annotated">{"".join(parts)}</div></div>'
        '<div class="card"><div class="card-head"><h3>Entity details</h3><span class="muted">Hover or focus a row to locate its mention</span></div>'
        '<div class="table-wrap"><table aria-label="Extracted entities"><thead><tr><th scope="col">Text</th><th scope="col">Label</th><th scope="col">Start</th><th scope="col">End</th><th scope="col">Method</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>{empty}</div>'
        '<p class="hint">Offsets refer to the displayed document. End positions are exclusive.</p>'
    ) + """
</section><script>
const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
// Commit the transparent first frame before starting background-color transitions.
requestAnimationFrame(()=>requestAnimationFrame(()=>document.body.classList.add('revealed')));
if (!reduced) {
 const counters = [...document.querySelectorAll('[data-count]')];
 counters.forEach(el => el.textContent = '0');
 const start = performance.now();
 function tick(now) {
   const progress = Math.min((now-start)/300,1);
   counters.forEach(el => el.textContent = Math.round(Number(el.dataset.count)*(1-Math.pow(1-progress,3))));
   if(progress<1) requestAnimationFrame(tick);
 }
 requestAnimationFrame(tick);
}
function link(id, active) {
 document.querySelectorAll('[data-id="'+id+'"]').forEach(el=>el.classList.toggle('linked',active));
}
document.querySelectorAll('[data-id]').forEach(el=>{
 el.addEventListener('mouseenter',()=>link(el.dataset.id,true));
 el.addEventListener('mouseleave',()=>link(el.dataset.id,false));
 el.addEventListener('focus',()=>link(el.dataset.id,true));
 el.addEventListener('blur',()=>link(el.dataset.id,false));
});
</script></body></html>"""


def main() -> None:
    st.set_page_config(page_title="MedExtract · Clinical text extraction", page_icon="🌊", layout="wide")
    try:
        with st.spinner("Preparing the extraction model…"):
            ensure_model()
    except Exception:
        logging.getLogger(__name__).exception("Model startup failed")
        st.error("The extraction model could not be prepared. Check the deployment logs and confirm that data/train/ contains the training notes and annotations.")
        st.stop()
    first_render = not st.session_state.get("_input_card_seen", False)
    entrance_css = '<style>.st-key-input_card { animation:input-rise 350ms ease-out both; }</style>' if first_render else ""
    st.markdown(OCEAN_CSS + entrance_css, unsafe_allow_html=True)
    st.session_state["_input_card_seen"] = True
    st.markdown('''<div class="brand"><span class="brand-icon" aria-hidden="true"><svg width="27" height="27" viewBox="0 0 28 28" fill="none"><path d="M3 14h6l3-7 4 14 3-7h6" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></span><div>MedExtract<div class="brand-subtitle">CLINICAL TEXT, CLARIFIED</div></div></div>''', unsafe_allow_html=True)

    st.markdown('''<div class="hero"><div class="eyebrow">Document intelligence</div><h1>Clinical notes, clearly structured.</h1><p>Find medicines, tests, symptoms, and dates in your notes.<br>Review every mention in context, then export the details.</p></div>''', unsafe_allow_html=True)
    with st.container(key="input_card"):
        st.markdown('<div class="card-heading">Add a clinical note</div><div class="card-description">Paste your text or choose a document to get started.</div>', unsafe_allow_html=True)
        mode = st.radio("Input source", ["Paste text", "Upload file"], horizontal=True, key="input_mode", on_change=clear_results, label_visibility="collapsed")
        if mode == "Paste text":
            text_input = st.text_area("Clinical note", height=225, key="note_text", on_change=clear_results,
                                      placeholder="Paracetamol helped headache. A complete blood count was requested on 15 January 2024.")
        else:
            # The native dropzone handles dragging, browsing, and keyboard access.
            # Its accessible label describes both input methods; CSS supplies the
            # centered visual prompt inside the target without duplicating it.
            uploaded_file = st.file_uploader("Drag & drop your file here, or click to browse", type=["txt", "pdf"],
                                              key="note_file", on_change=clear_results, label_visibility="collapsed")
            st.caption("TXT or text-based PDF. Scanned documents and handwriting are not supported.")
        button_slot = st.empty()
        extract_clicked = button_slot.button("Extract →", type="primary", use_container_width=True, key="extract")

    if extract_clicked:
        clear_results()
        button_slot.button("Extracting…", type="primary", disabled=True, use_container_width=True, key="extract_busy")
        try:
            with st.spinner("Reading your note and locating entities…"):
                if mode == "Upload file":
                    if uploaded_file is None:
                        st.session_state.notice = ("info", "Choose a TXT or PDF document, then select Extract.")
                        text = ""
                    else:
                        text = read_uploaded_file(uploaded_file)
                else:
                    text = text_input
                if not text.strip():
                    if "notice" not in st.session_state:
                        message = "The document contains no readable text. Try a TXT file or a text-based PDF." if st.session_state.input_mode == "Upload file" else "Paste a clinical note, then select Extract."
                        st.session_state.notice = ("info", message)
                else:
                    st.session_state.result = {"text": text, "entities": extract_entities(text)}
        except InputProcessingError:
            st.session_state.notice = ("error", "This document could not be read. Check that it is a valid, readable TXT or PDF file and try again.")
        except FileNotFoundError:
            st.session_state.notice = ("error", "The extraction model is unavailable. Restore the trained model and try again.")
        except Exception:
            st.session_state.notice = ("error", "Extraction could not be completed. Please try again with a shorter note or another document.")
        st.rerun()

    if "notice" in st.session_state:
        kind, message = st.session_state.notice
        getattr(st, kind)(message)
    if "result" in st.session_state:
        result = st.session_state.result
        # Keep short results compact while bounding long documents and tables.
        panel_height = min(920, 380 + min(240, 55 + result["text"].count("\n") * 30
                                        + len(result["text"]) // 80 * 30)
                           + min(305, len(result["entities"]) * 44))
        components.html(results_html(result["text"], result["entities"]), height=panel_height, scrolling=True)
        st.download_button("Download JSON ↓", data=json.dumps(result["entities"], ensure_ascii=False, indent=2),
                           file_name="medextract_entities.json", mime="application/json", use_container_width=False)
    else:
        st.markdown('<div class="ready-state"><strong>Ready when you are</strong><p>Your extracted entities and annotated text will appear here.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="footer-note">MedExtract · Medical document information extraction</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
