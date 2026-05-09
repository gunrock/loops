#!/usr/bin/env python3
"""Build script for the loops project website.

Assembles the static site from:
  - Jinja2 templates (site/templates/)
  - Markdown content (site/content/ and docs/)
  - Doxygen XML output (build/doxygen-xml/)
  - Static assets (site/static/)

Output: _site/ (flat directory, ready for GitHub Pages)
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown
import yaml
from jinja2 import Environment, FileSystemLoader
from lxml import etree

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
OUT = ROOT / "_site"
TEMPLATES = SITE / "templates"
CONTENT = SITE / "content"
STATIC = SITE / "static"
DOCS = ROOT / "docs"
DOXYGEN_XML = ROOT / "_doxygen" / "xml"

BASE_URL = os.environ.get("BASE_URL", "/loops").rstrip("/")


def clean():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)


def copy_static():
    shutil.copytree(STATIC, OUT / "static")


def get_jinja_env():
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=False)
    env.globals["base"] = BASE_URL
    return env


def parse_frontmatter(text):
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            meta = yaml.safe_load(parts[1])
            body = parts[2]
            return meta or {}, body
    return {}, text


def render_markdown(text):
    md = markdown.Markdown(
        extensions=["fenced_code", "tables", "codehilite", "toc"],
        extension_configs={
            "codehilite": {"css_class": "highlight", "guess_lang": False},
            "toc": {"permalink": False},
        },
    )
    html = md.convert(text)
    toc_items = []
    for m in re.finditer(r'<h([23])\s+id="([^"]+)"[^>]*>(.+?)</h\1>', html):
        toc_items.append({"id": m.group(2), "text": re.sub(r"<[^>]+>", "", m.group(3))})
    html = wrap_code_blocks(html)
    html = html.replace('href="/README.md"', f'href="{BASE_URL}/"')
    html = re.sub(r'href="/README\.md#[^"]*"', f'href="{BASE_URL}/docs/getting-started/"', html)
    html = re.sub(
        r'href="\.\./([^"]+)"',
        r'href="https://github.com/gunrock/loops/blob/main/\1"',
        html,
    )
    return html, toc_items


def wrap_code_blocks(html):
    def replacer(m):
        inner = m.group(0)
        lang_m = re.search(r'class="[^"]*language-(\w+)', inner)
        lang = lang_m.group(1) if lang_m else ""
        header = ""
        if lang:
            header = f'<div class="code-header"><span class="code-lang">{lang}</span><button class="code-copy">Copy</button></div>'
        return f'<div class="code-block">{header}{inner}</div>'

    return re.sub(r"<div class=\"highlight\">.*?</div>\s*</div>", replacer, html, flags=re.DOTALL)


def build_markdown_pages(env):
    pages = []

    # Site content
    for md_file in sorted(CONTENT.rglob("*.md")):
        rel = md_file.relative_to(CONTENT)
        pages.append((md_file, rel))

    # Existing docs
    doc_map = {
        "build.md": Path("build.md"),
        "experimentation.md": Path("experimentation.md"),
        "datasets.md": Path("datasets.md"),
        "reproducing-results.md": Path("reproducing-results.md"),
    }
    for filename, rel in doc_map.items():
        src = DOCS / filename
        if src.exists():
            pages.append((src, rel))

    template = env.get_template("page.html")

    for src, rel in pages:
        text = src.read_text()
        meta, body = parse_frontmatter(text)
        html, toc_items = render_markdown(body)

        # Determine output path
        stem = rel.stem
        if "concepts" in str(rel):
            out_dir = OUT / "docs" / "concepts" / stem
        elif src.parent == DOCS:
            out_dir = OUT / "docs" / stem
        else:
            out_dir = OUT / "docs" / stem

        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "index.html"

        rendered = template.render(
            title=meta.get("title", stem.replace("-", " ").title()),
            description=meta.get("description", ""),
            content=html,
            toc_items=toc_items,
        )
        out_file.write_text(rendered)
        print(f"  page: /docs/{stem}/ -> {out_file.relative_to(OUT)}")


def build_landing(env):
    src = TEMPLATES / "landing.html"
    content = src.read_text()
    if BASE_URL:
        content = content.replace('href="/', f'href="{BASE_URL}/')
        content = content.replace("href='/", f"href='{BASE_URL}/")
        content = content.replace('src="/', f'src="{BASE_URL}/')
        content = content.replace("src='/", f"src='{BASE_URL}/")
        content = content.replace("url('/", f"url('{BASE_URL}/")
        content = content.replace('url("/', f'url("{BASE_URL}/')
    out = OUT / "index.html"
    out.write_text(content)
    print(f"  landing: / -> index.html")


def run_doxygen():
    doxyfile = SITE / "Doxyfile"
    if not doxyfile.exists():
        print("  WARN: No Doxyfile found, skipping doxygen")
        return False
    print("  Running doxygen...")
    result = subprocess.run(
        ["doxygen", str(doxyfile)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  WARN: doxygen exited with {result.returncode}")
        print(result.stderr[:500])
        return False
    return DOXYGEN_XML.exists()


def parse_doxygen_xml():
    if not DOXYGEN_XML.exists():
        return [], [], []

    containers = []
    layouts = []
    schedules = []

    index = DOXYGEN_XML / "index.xml"
    if not index.exists():
        return containers, layouts, schedules

    tree = etree.parse(str(index))
    for compound in tree.findall(".//compound"):
        kind = compound.get("kind")
        if kind not in ("class", "struct"):
            continue
        name = compound.findtext("name", "")
        refid = compound.get("refid", "")

        if not name.startswith("loops::"):
            continue

        brief = ""
        detail_file = DOXYGEN_XML / f"{refid}.xml"
        tparams = []
        members_list = []
        methods_list = []
        file_path = ""
        detailed = ""

        if detail_file.exists():
            dtree = etree.parse(str(detail_file))
            bd = dtree.find(".//compounddef/briefdescription")
            if bd is not None:
                brief = "".join(bd.itertext()).strip()
            dd = dtree.find(".//compounddef/detaileddescription")
            if dd is not None:
                texts = []
                for para in dd.findall("para"):
                    if para.find("parameterlist") is not None:
                        continue
                    t = "".join(para.itertext()).strip()
                    if t:
                        texts.append(t)
                detailed = " ".join(texts)[:300] if texts else ""

            loc = dtree.find(".//location")
            if loc is not None:
                f = loc.get("file", "")
                if f:
                    file_path = f

            for tp in dtree.findall(".//templateparamlist/param"):
                tp_name = ""
                tp_desc = ""
                defname = tp.find("defname")
                declname = tp.find("declname")
                tp_type = tp.find("type")
                if defname is not None and defname.text:
                    tp_name = defname.text
                elif declname is not None and declname.text:
                    tp_name = declname.text
                elif tp_type is not None and tp_type.text:
                    parts = tp_type.text.strip().split()
                    tp_name = parts[-1] if parts else tp_type.text.strip()
                if tp_name:
                    tparams.append({"name": tp_name, "desc": tp_desc})

            for pitem in dtree.findall(
                ".//detaileddescription//parameterlist[@kind='templateparam']/parameteritem"
            ):
                pname = pitem.findtext(".//parametername", "").strip()
                pdesc_el = pitem.find(".//parameterdescription")
                pdesc = "".join(pdesc_el.itertext()).strip() if pdesc_el is not None else ""
                for tp in tparams:
                    if tp["name"] == pname and pdesc:
                        tp["desc"] = pdesc
                        break

            for member in dtree.findall(".//memberdef"):
                mk = member.get("kind", "")
                mname = member.findtext("name", "")
                if mname.startswith("_") or mk == "friend":
                    continue
                mbr = member.find("briefdescription")
                mdesc = "".join(mbr.itertext()).strip() if mbr is not None else ""
                if mk == "function":
                    ret = member.findtext("type", "")
                    args = member.findtext("argsstring", "")
                    methods_list.append({
                        "name": mname,
                        "returns": ret,
                        "args": args.strip("()"),
                        "detail": mdesc,
                    })
                elif mk == "variable":
                    members_list.append({"name": mname, "desc": mdesc})

        if "<" in name:
            continue

        short_name = name.replace("loops::", "")
        slug = short_name.replace("::", "/")
        url = f"{BASE_URL}/api/loops/{slug}/"

        entry = {
            "name": short_name,
            "kind": kind,
            "brief": brief or f"{short_name} type.",
            "url": url,
            "detailed": detailed,
            "tparams": tparams,
            "members": members_list,
            "methods": methods_list,
            "file": file_path,
            "refid": refid,
        }

        if "layout::" in name:
            layouts.append(entry)
        elif "schedule::" in name:
            schedules.append(entry)
        else:
            containers.append(entry)

    return containers, layouts, schedules


def build_api(env, containers, layouts, schedules):
    # API index
    index_tmpl = env.get_template("api_index.html")
    api_dir = OUT / "api"
    api_dir.mkdir(parents=True, exist_ok=True)

    rendered = index_tmpl.render(
        title="API Reference",
        containers=containers,
        layouts=layouts,
        schedules=schedules,
        toc_items=[],
    )
    (api_dir / "index.html").write_text(rendered)
    print(f"  api: /api/ -> api/index.html")

    # Individual API pages
    api_tmpl = env.get_template("api.html")
    for items in [containers, layouts, schedules]:
        for item in items:
            slug = item["name"].replace("::", "/").split("<")[0]
            page_dir = api_dir / "loops" / slug
            page_dir.mkdir(parents=True, exist_ok=True)
            rendered = api_tmpl.render(
                title=item["name"],
                name=item["name"],
                kind=item["kind"],
                brief=item["brief"],
                detailed=item.get("detailed", ""),
                file=item.get("file", ""),
                tparams=item.get("tparams", []),
                members=item.get("members", []),
                methods=item.get("methods", []),
                toc_items=[],
            )
            (page_dir / "index.html").write_text(rendered)
            print(f"  api: /api/loops/{slug}/ -> api/loops/{slug}/index.html")


def main():
    print("Building loops website...")
    clean()
    copy_static()
    (OUT / ".nojekyll").touch()
    print("Static assets copied.")

    env = get_jinja_env()

    print("Building pages...")
    build_landing(env)
    build_markdown_pages(env)

    print("Running doxygen...")
    has_doxygen = run_doxygen()

    print("Building API reference...")
    containers, layouts, schedules = parse_doxygen_xml()
    build_api(env, containers, layouts, schedules)

    print(f"\nDone. Site written to {OUT}")
    print(f"  {sum(1 for _ in OUT.rglob('*.html'))} HTML pages")
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"  {total / 1024:.0f} KB total")


if __name__ == "__main__":
    main()
