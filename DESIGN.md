# DESIGN.md — loops visual identity & website specification

> **Codename:** The Grid Dissolves
>
> A complete design language for the `loops` GPU load-balancing framework.
> This document specifies every visual decision — from color values to
> animation curves — needed to build the project website without
> further design input.

---

## 1. Design Concept

The website tells the story of `loops` in a single scroll:

1. **A perfect grid** — the GPU's ideal: uniform, regular, every cell equal.
2. **The grid breaks** — real workloads are irregular; cells swell, shrink,
   cluster, leave gaps. This is the problem.
3. **The grid rebalances** — `loops` redistributes atoms across tiles,
   and the grid settles into a new, balanced arrangement. This is the
   solution.

Every visual choice descends from this narrative. The site is dark because
GPUs live in server racks, not sunlit offices. The accent color is singular
because the library's thesis is singular: one abstraction that unifies
many schedules. Whitespace is generous because the code is dense and
deserves room to breathe.

---

## 2. Color System

### 2.1 Core palette

```
Token                  Hex        RGB               Usage
─────────────────────────────────────────────────────────────
--bg-primary           #07070D    rgb(7, 7, 13)     Page background
--bg-elevated          #0F0F1A    rgb(15, 15, 26)   Cards, code blocks, nav
--bg-surface           #161625    rgb(22, 22, 37)   Hover states, borders
--bg-subtle            #1E1E32    rgb(30, 30, 50)   Active states, selection

--text-primary         #E8E8F0    rgb(232, 232, 240) Body text
--text-secondary       #9090A8    rgb(144, 144, 168) Captions, metadata
--text-tertiary        #5A5A72    rgb(90, 90, 114)   Disabled, decorative

--accent               #6366F1    rgb(99, 102, 241)  Primary accent (indigo)
--accent-hover         #818CF8    rgb(129, 140, 248)  Accent hover / light
--accent-muted         #6366F133  rgba(99,102,241,.2) Accent backgrounds
--accent-glow          #6366F140  rgba(99,102,241,.25) Code block glow

--border               #1E1E32    rgb(30, 30, 50)   Subtle borders
--border-accent        #6366F166  rgba(99,102,241,.4) Highlighted borders

--success              #10B981    rgb(16, 185, 129)  Passing tests, valid
--warning              #F59E0B    rgb(245, 158, 11)  Deprecation notices
--error                #EF4444    rgb(239, 68, 68)   Errors, breaking changes
```

### 2.2 Syntax highlighting (code blocks)

Built on the same palette. No rainbow; restrained and readable.

```
Token                  Hex        Scope
─────────────────────────────────────────────────────────────
--syn-keyword          #818CF8    keywords, storage types
--syn-type             #E8E8F0    type names, classes, structs
--syn-function         #C4B5FD    function/method names
--syn-string           #6EE7B7    string literals
--syn-number           #FCD34D    numeric literals
--syn-comment          #5A5A72    comments
--syn-operator         #9090A8    operators, punctuation
--syn-preprocessor     #F9A8D4    preprocessor directives
--syn-template-param   #93C5FD    template parameters
```

### 2.3 Diagram palette

Used for animated and static diagrams illustrating scheduling concepts.

```
--diagram-tile         #6366F1    Tile boundaries (rows, columns, blocks)
--diagram-atom         #C4B5FD    Individual atoms (nonzeros, elements)
--diagram-processor    #10B981    Processor/thread assignments
--diagram-imbalanced   #EF4444    Overloaded threads (before balancing)
--diagram-balanced     #6366F1    Balanced threads (after)
--diagram-grid         #1E1E32    Background grid lines
```

### 2.4 Usage rules

- **One accent color.** Indigo (`#6366F1`) is the only chromatic color in
  the UI. It marks interactive elements, the current page in navigation,
  and the "after balancing" state in diagrams.
- **Semantic colors** (success/warning/error) appear only in functional
  contexts (build badges, API stability markers), never decoratively.
- **No gradients** in UI elements. Gradients are reserved exclusively for
  the hero animation and the faint glow behind code blocks.
- **Contrast:** `--text-primary` on `--bg-primary` yields a ratio of
  ~17:1, exceeding WCAG AAA. `--text-secondary` on `--bg-primary` yields
  ~5.5:1, meeting WCAG AA.

---

## 3. Typography

### 3.1 Type stack

```
Role           Font              Fallback                   Weight
───────────────────────────────────────────────────────────────────
Code           JetBrains Mono    'Fira Code', monospace     400, 700
Headings       Space Grotesk     'Inter', sans-serif        500, 700
Body           Inter             system-ui, sans-serif      400, 500
```

All three are variable-weight, open-source Google Fonts. Self-hosted via
the site build to avoid third-party requests.

### 3.2 Type scale

Desktop base: `16px`. Scale factor: `1.25` (major third).

```
Token          Size      Line-height   Letter-spacing   Usage
──────────────────────────────────────────────────────────────
--text-xs      12.8px    1.5           +0.02em          Badges, fine print
--text-sm      14px      1.5           +0.01em          Captions, metadata
--text-base    16px      1.7           0                Body text
--text-lg      20px      1.6           -0.01em          Lead paragraphs
--text-xl      25px      1.4           -0.015em         H3, section titles
--text-2xl     31.25px   1.3           -0.02em          H2, page titles
--text-3xl     39px      1.2           -0.025em         H1, hero headline
--text-4xl     48.8px    1.1           -0.03em          Display (hero only)
```

### 3.3 Code typography

```css
.code-block {
  font-family: 'JetBrains Mono', monospace;
  font-size: 14px;
  line-height: 1.7;
  tab-size: 2;
  font-variant-ligatures: contextual;   /* fi, fl only; no => arrows */
}
```

- Line numbers: `--text-tertiary`, right-aligned, `font-variant-numeric:
  tabular-nums`.
- File path header: `--text-secondary`, `--text-xs`, rendered above the
  code block with a subtle top border in `--border`.
- Copy button: appears on hover, top-right, `--text-tertiary` → `--accent`
  on hover.

### 3.4 Responsive scaling

At `< 768px`, the base drops to `15px` and the scale factor tightens to
`1.2` (minor third). Code blocks use `13px` and gain horizontal scroll
with a faded edge mask.

---

## 4. Layout & Grid

### 4.1 Page grid

```
Desktop (≥ 1280px):
┌──────────────────────────────────────────────────┐
│ 240px sidebar │ 720px content │ 200px TOC        │
│ (nav)         │ (main)        │ (on-this-page)   │
└──────────────────────────────────────────────────┘
Max content width: 1200px, centered.

Tablet (768–1279px):
Sidebar collapses to hamburger. TOC becomes a sticky dropdown.
Content fills available width with 32px side padding.

Mobile (< 768px):
Single column. 20px side padding.
Sidebar is a full-screen overlay. TOC is hidden (scroll-to-top button).
```

### 4.2 Content grid

Within the 720px content column, a 12-column micro-grid with 16px
gutters handles mixed-width elements:

- **Prose**: columns 1–10 (600px max measure for readability).
- **Code blocks**: columns 1–12 (full width, may overflow into TOC
  gutter on desktop for long lines).
- **Diagrams**: columns 1–12.
- **Tables**: columns 1–12, horizontal scroll on overflow.
- **Callouts/notes**: columns 1–10, left border in `--accent`.

### 4.3 Spacing scale

Based on an 8px unit:

```
--space-1    4px       Inline padding, icon gaps
--space-2    8px       Tight padding
--space-3    12px      Default padding
--space-4    16px      Section element gaps
--space-5    24px      Between content blocks
--space-6    32px      Between sections
--space-7    48px      Between major page regions
--space-8    64px      Hero padding, page top/bottom
--space-9    96px      Landing page section spacing
```

---

## 5. Components

### 5.1 Navigation sidebar

```
┌─────────────────────┐
│  ◌ loops             │  Logo + wordmark, links to /
│                     │
│  Getting Started    │  Section headers: --text-secondary, uppercase,
│    Installation     │  --text-xs, --space-4 top margin, --space-2 bottom.
│    Quick Start      │
│    Sanity Check     │  Links: --text-primary, 14px.
│                     │  Active link: --accent, left border 2px --accent.
│  Concepts           │  Hover: --text-primary → --accent-hover.
│    Abstraction      │
│    Tiles & Atoms    │
│    Schedules        │
│                     │
│  API Reference      │  Expandable sections with caret icon.
│    ▸ Containers     │  Doxygen-generated pages linked here.
│    ▸ Layouts        │
│    ▸ Schedules      │
│    ▸ Utilities      │
│                     │
│  Examples           │
│  Benchmarks         │
│  Paper              │  Links to DOI
│                     │
│  ─────────────────  │  Divider: 1px --border
│  GitHub ↗           │  External links: trailing arrow icon
│  Cite ↗             │
└─────────────────────┘
```

Background: `--bg-elevated`. Border-right: 1px `--border`.
Sticky, full viewport height, scrollable independently.

### 5.2 Code blocks

```
┌─ include/loops/schedule.hxx ────────────────────── ⧉ ┐
│                                                      │
│  1 │ using setup_t = schedule::setup<                │
│  2 │     schedule::algorithms_t::thread_mapped,      │
│  3 │     128, 1, index_t, offset_t>;                 │
│  4 │                                                 │
│  5 │ setup_t config(offsets, rows, nnzs);            │
│  6 │                                                 │
│  7 │ for (auto row : config.tiles()) {               │
│  8 │   for (auto nz : config.atoms(row)) {           │
│  9 │     y[row] += values[nz] * x[indices[nz]];     │
│ 10 │   }                                             │
│ 11 │ }                                               │
│                                                      │
└──────────────────────────────────────────────────────┘
```

- Background: `--bg-elevated`.
- Border: 1px `--border`, `border-radius: 6px`.
- Subtle box-shadow: `0 0 24px var(--accent-glow)` — a faint indigo halo.
- File path header: sticky within the code block if it scrolls.
- Line highlighting: selected lines get `--accent-muted` background.
- Language badge: bottom-right, `--text-tertiary`, `--text-xs`.

### 5.3 Callout blocks

Four variants, each with a left border and matching icon:

```
Note     │  --accent       │  ℹ  info circle
Tip      │  --success      │  ✦  sparkle
Warning  │  --warning      │  ▲  triangle
Danger   │  --error        │  ●  filled circle
```

Background: the variant color at 8% opacity. Text: `--text-primary`.
Title: variant color, `--text-sm`, `font-weight: 500`.

### 5.4 API cards (doxygen-generated)

Each class/struct/function gets a card:

```
┌──────────────────────────────────────────────────────┐
│  struct  csr_t<index_t, offset_t, value_t, space>    │  ← type badge + name
│                                                      │
│  Compressed Sparse Row format. Owns row offsets,     │  ← brief (from @brief)
│  column indices, and values arrays.                  │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │  csr_t(r, c, nnz)           constructor        │  │  ← member list
│  │  rows, cols, nnzs           dimensions          │  │
│  │  offsets, indices, values   storage              │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  Defined in  loops/container/csr.hxx                 │  ← source link
└──────────────────────────────────────────────────────┘
```

- Card background: `--bg-elevated`, `border-radius: 8px`.
- Type badge: `struct` / `class` / `enum` / `function` — small pill,
  `--accent-muted` background, `--accent` text.
- Member list: alternating `--bg-primary` / `--bg-elevated` rows.
- Source link: `--text-tertiary`, links to GitHub blob.

### 5.5 The three-domain diagram

The project's architectural diagram (data / schedule / computation) is a
first-class component, not an image. Built in SVG, it appears on the
landing page and in the Concepts section.

```
     DATA                 SCHEDULE              COMPUTATION
  ┌─────────┐         ┌─────────────┐         ┌─────────────┐
  │ csr_t   │         │ thread      │         │ for (tile)   │
  │ coo_t   │  ────▸  │ group       │  ────▸  │   for (atom) │
  │ ell_t   │         │ work        │         │     y += ... │
  │ bcsr_t  │         │ merge_path  │         │              │
  │ dia_t   │         │             │         │              │
  └─────────┘         └─────────────┘         └─────────────┘
      tiles                layout                  kernel
     + atoms              contract
```

In the SVG version:
- Data column: `--diagram-tile` borders, `--diagram-atom` dots.
- Schedule column: `--accent` borders.
- Computation column: `--text-primary` monospace text.
- Arrows: animated dashed lines in `--accent`, 2px stroke.

### 5.6 Tables

```css
table {
  width: 100%;
  border-collapse: collapse;
}
th {
  text-align: left;
  color: var(--text-secondary);
  font-size: var(--text-sm);
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding: var(--space-3) var(--space-4);
  border-bottom: 1px solid var(--border);
}
td {
  padding: var(--space-3) var(--space-4);
  border-bottom: 1px solid var(--border);
  color: var(--text-primary);
}
tr:hover td {
  background: var(--bg-surface);
}
```

### 5.7 Buttons and links

- **Primary button**: `--accent` background, white text, `border-radius:
  6px`, `padding: 10px 20px`. Hover: `--accent-hover`. Used sparingly
  (GitHub link, "Get Started").
- **Text links**: `--accent` underline on hover only. No underline at rest
  — the indigo color is sufficient contrast against `--text-primary`.
- **External links**: trailing `↗` icon, `--text-secondary`.

---

## 6. Logo & Wordmark

### 6.1 Concept

The logo is an abstract representation of a loop (the load-balancing
feedback cycle) rendered as a continuous geometric trace. It references
both the `for` loop in code and the redistribution loop of atoms → tiles
→ processors → rebalance.

### 6.2 Construction

```
    ◌
   ╱ ╲
  ╱   ╲      A rounded square with one corner left open,
 │     │     suggesting both a loop and a bracket (code).
  ╲   ╱      The open corner faces top-right (forward motion).
   ╲ ╱
    ·
```

- Stroke only, no fill. Stroke width: proportional to the glyph height.
- Rendered in `--accent` on dark backgrounds, `--bg-primary` on light.
- Minimum clear space: 1× the logo height on all sides.

### 6.3 Wordmark

`loops` set in **Space Grotesk Medium**, all lowercase, tracked at
`-0.03em`. The wordmark always appears to the right of the logo mark,
vertically centered. On the website nav, the logo mark is `20px` tall
and the wordmark is `16px`.

### 6.4 Favicon

The logo mark simplified to a single-color glyph, rendered at `32×32`
and `16×16`. Background: `--bg-primary`. Stroke: `--accent`.

---

## 7. Landing Page

### 7.1 Structure

```
┌──────────────────────────────────────────────────────────┐
│                         NAV BAR                          │
│  ◌ loops     Docs   API   Examples   GitHub ↗            │
├──────────────────────────────────────────────────────────┤
│                                                          │
│                    HERO SECTION                           │
│                                                          │
│           Expressing Parallel                            │
│           Irregular Computations                         │
│                                                          │
│  A GPU load-balancing framework that decouples           │
│  scheduling from computation. Write the kernel once,     │
│  swap schedules freely.                                  │
│                                                          │
│  [ Get Started ]  [ View on GitHub ↗ ]                   │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │          ◆ THE GRID DISSOLVES ANIMATION ◆          │  │
│  │  (described in §8 — the animated hero canvas)      │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ── THE PROBLEM ──────────────────────────────────────   │
│                                                          │
│  [Animated diagram: naive thread mapping showing         │
│   imbalanced workload — some threads idle, some          │
│   overloaded. --diagram-imbalanced highlights.]          │
│                                                          │
│  Irregular data structures — sparse matrices, power-law  │
│  graphs, variable-length lists — don't map well to the   │
│  GPU's regular architecture.                             │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ── THE ABSTRACTION ──────────────────────────────────   │
│                                                          │
│  [Three-domain SVG: DATA → SCHEDULE → COMPUTATION]       │
│                                                          │
│  Three concerns, separated:                              │
│                                                          │
│  ┌────────┐  ┌──────────┐  ┌──────────────┐             │
│  │  DATA  │  │ SCHEDULE │  │ COMPUTATION  │             │
│  │        │  │          │  │              │             │
│  │ Choose │  │ Choose a │  │ Write your   │             │
│  │ your   │  │ balancing│  │ kernel once. │             │
│  │ format.│  │ strategy.│  │ It works     │             │
│  │        │  │          │  │ with all     │             │
│  │ CSR    │  │ thread   │  │ schedules.   │             │
│  │ COO    │  │ group    │  │              │             │
│  │ ELL    │  │ work     │  │ for (tile)   │             │
│  │ BCSR   │  │ merge    │  │  for (atom)  │             │
│  │ DIA    │  │          │  │   compute()  │             │
│  └────────┘  └──────────┘  └──────────────┘             │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ── CODE ─────────────────────────────────────────────   │
│                                                          │
│  A tabbed code block showing the same SpMV kernel with   │
│  different schedules — the user computation stays the    │
│  same, only the schedule::setup<...> line changes.       │
│                                                          │
│  Tabs: [ thread_mapped | work_oriented | merge_path ]    │
│                                                          │
│  [Code block per §5.2]                                   │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ── FORMAT-GENERIC LAYOUTS ───────────────────────────   │
│                                                          │
│  The layout contract table from the README, rendered     │
│  per §5.6 table styling. Each row links to the API       │
│  reference card for that layout.                         │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ── CITE ─────────────────────────────────────────────   │
│                                                          │
│  PPoPP 2023 citation in a styled code block with a       │
│  one-click copy button. DOI badge linking to the paper.  │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  FOOTER                                                  │
│  Apache-2.0 · UC Davis · Gunrock · GitHub                │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 7.2 Nav bar

- Transparent over the hero, transitioning to `--bg-elevated` with a
  `backdrop-filter: blur(12px)` once the user scrolls past the hero.
- Height: `56px`. Sticky.
- Items: `--text-secondary`, hover → `--text-primary`.
  Active: `--accent`, `font-weight: 500`.

---

## 8. Motion & Animation

### 8.1 The hero animation ("The Grid Dissolves")

A `<canvas>` element (or WebGL for performance) behind the hero text.

**Sequence** (loops continuously, ~12s cycle):

1. **Regular grid** (0–3s): A 20×20 grid of small squares, uniformly
   spaced, all `--border` colored. Each square represents a "cell" of
   GPU work. Subtle idle drift (±1px, sine wave) to keep it alive.

2. **Dissolution** (3–6s): Squares begin to resize unevenly — some grow
   3×, some shrink to dots, some disappear. Gaps form. Clusters emerge.
   The grid is now "irregular." Color shifts: oversized cells pick up
   `--diagram-imbalanced` tint. Empty cells fade to `--bg-primary`.
   Easing: `cubic-bezier(0.4, 0, 0.2, 1)`.

3. **Rebalancing** (6–9s): A subtle wave sweeps left-to-right
   (representing the load-balancing pass). As it crosses each column,
   cells redistribute — the oversized ones split, the gaps fill. The
   grid doesn't return to perfect uniformity; it settles into a
   *balanced* irregularity where every cell is within ±20% of the mean.
   Color: all cells transition to `--accent` as they balance. Easing:
   `cubic-bezier(0, 0, 0.2, 1)`.

4. **Hold** (9–12s): The balanced grid holds, gently pulsing. Then it
   fades back to the regular grid and the cycle restarts.

**Performance:**
- Uses `requestAnimationFrame`, respects `prefers-reduced-motion` (falls
  back to a static balanced-grid illustration).
- Canvas resolution: `devicePixelRatio`-aware, capped at 2×.
- Target: 60fps on integrated graphics; degrade gracefully by reducing
  grid size on low-power devices.

### 8.2 Scroll-triggered diagrams

Diagrams in the "Problem" and "Abstraction" sections animate on
`IntersectionObserver` entry:

- **Problem diagram**: threads appear one by one (staggered 30ms), then
  work items distribute unevenly, revealing the imbalance.
- **Abstraction diagram**: the three columns draw in left-to-right,
  connected by animated dashed arrows.

### 8.3 Micro-interactions

- **Page transitions**: none. The site is a multi-page static site; each
  page loads fresh. No SPA overhead.
- **Link hover**: underline slides in from left, 200ms ease-out.
- **Code block hover**: the faint `--accent-glow` box-shadow intensifies
  slightly (0.25 → 0.35 opacity), 150ms.
- **Sidebar active indicator**: the 2px left border grows from center
  outward, 150ms.
- **Copy button**: on click, icon morphs from clipboard → checkmark,
  holds 1.5s, morphs back.

### 8.4 Motion tokens

```
--duration-fast      100ms
--duration-normal    200ms
--duration-slow      400ms
--ease-default       cubic-bezier(0.4, 0, 0.2, 1)
--ease-out           cubic-bezier(0, 0, 0.2, 1)
--ease-spring        cubic-bezier(0.34, 1.56, 0.64, 1)
```

---

## 9. Page Architecture

### 9.1 Page inventory

```
Path                          Source               Generator
────────────────────────────────────────────────────────────────
/                             site/index.html      static (hand-authored)
/docs/getting-started/        site/docs/*.md       static site generator
/docs/concepts/abstraction/   site/docs/*.md       static site generator
/docs/concepts/tiles-atoms/   site/docs/*.md       static site generator
/docs/concepts/schedules/     site/docs/*.md       static site generator
/docs/build/                  docs/build.md        static site generator
/docs/experimentation/        docs/experimentation.md  static site generator
/docs/datasets/               docs/datasets.md     static site generator
/docs/reproducing-results/    docs/reproducing-results.md  static site generator
/api/                         doxygen XML → HTML   doxygen + custom theme
/api/loops/csr_t/             doxygen XML → HTML   doxygen + custom theme
/api/loops/schedule/setup/    doxygen XML → HTML   doxygen + custom theme
/api/loops/layout/csr/        doxygen XML → HTML   doxygen + custom theme
  ... (one page per documented symbol)
/examples/                    examples/**/*.cu     static site generator
/benchmarks/                  plots/               static site generator
```

### 9.2 Static site generator

**Tool: custom lightweight build** using a shell/Python pipeline:

1. Markdown → HTML via `pandoc` (or a Python markdown library with
   fenced-code-block and table extensions).
2. Templates: plain HTML/CSS/JS — no framework. A single `base.html`
   template with `{{ content }}`, `{{ title }}`, `{{ toc }}` slots.
3. Syntax highlighting: server-side via `pygments` (Monokai-derived theme
   matching §2.2 tokens) or a build-time `highlight.js` pass.
4. Output: flat `_site/` directory, ready for GitHub Pages.

Why not Hugo/Jekyll/Docusaurus: the design is custom enough that a
framework's opinions would fight it. The site has ~20 pages, not 2000.
A 200-line Python build script is simpler than configuring around a
framework's layout system.

### 9.3 Doxygen integration

Doxygen generates **XML output** (not HTML). A post-processing step
transforms doxygen XML into pages that match the site's design:

```
Pipeline:
  Doxyfile (XML output)
      │
      ▼
  doxygen → build/doxygen-xml/
      │
      ▼
  scripts/doxygen_to_html.py
      │  - Reads XML, extracts: classes, structs, functions, enums,
      │    namespaces, briefs, detailed docs, member lists, source refs.
      │  - Renders into the site's base.html template.
      │  - Generates API cards per §5.4.
      │  - Builds the sidebar nav tree for the API section.
      │
      ▼
  _site/api/**/*.html
```

**Doxyfile key settings:**

```
PROJECT_NAME           = loops
OUTPUT_DIRECTORY       = build/doxygen-xml
GENERATE_XML           = YES
GENERATE_HTML          = NO
GENERATE_LATEX         = NO
XML_PROGRAMLISTING     = YES
INPUT                  = include/loops
RECURSIVE              = YES
FILE_PATTERNS          = *.hxx *.cuh
EXTRACT_ALL            = NO
EXTRACT_PRIVATE        = NO
EXTRACT_STATIC         = YES
JAVADOC_AUTOBRIEF      = YES
BUILTIN_STL_SUPPORT    = YES
EXCLUDE_PATTERNS       = */detail/*
```

This approach means:
- API docs are **always in sync** with the source (generated on every
  deploy).
- The API pages look identical to the rest of the site (same nav, same
  typography, same code blocks).
- No doxygen CSS to override — we never generate doxygen's HTML.

---

## 10. Build & Deploy Pipeline

### 10.1 GitHub Actions workflow

```yaml
# .github/workflows/pages.yml
name: Deploy Site

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y doxygen python3-pip
          pip3 install jinja2 markdown pygments pyyaml lxml

      - name: Generate doxygen XML
        run: doxygen site/Doxyfile

      - name: Build site
        run: python3 site/build.py

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: _site

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

### 10.2 Directory structure (new files)

```
site/
├── build.py                  # Build script: markdown → HTML, doxygen XML → API pages
├── Doxyfile                  # Doxygen configuration (XML output only)
├── templates/
│   ├── base.html             # Root template (nav, sidebar, footer)
│   ├── page.html             # Documentation page template
│   ├── api.html              # API reference page template
│   ├── api_index.html        # API overview / namespace listing
│   └── landing.html          # Landing page template (hero, sections)
├── static/
│   ├── css/
│   │   ├── tokens.css        # CSS custom properties (§2, §3, §4)
│   │   ├── base.css          # Reset, typography, global styles
│   │   ├── layout.css        # Grid, sidebar, responsive breakpoints
│   │   ├── components.css    # Code blocks, callouts, cards, tables
│   │   └── syntax.css        # Code syntax highlighting theme
│   ├── js/
│   │   ├── hero-canvas.js    # The Grid Dissolves animation
│   │   ├── sidebar.js        # Mobile toggle, active state tracking
│   │   ├── copy-code.js      # Copy-to-clipboard for code blocks
│   │   └── toc.js            # On-this-page scroll spy
│   ├── fonts/
│   │   ├── JetBrainsMono-*.woff2
│   │   ├── SpaceGrotesk-*.woff2
│   │   └── Inter-*.woff2
│   └── img/
│       ├── logo.svg          # Logo mark
│       ├── favicon.svg       # Favicon (SVG for modern browsers)
│       ├── favicon.ico       # Favicon (ICO fallback)
│       └── og-image.png      # Open Graph preview (1200×630)
├── content/
│   ├── index.md              # Landing page content blocks (rendered into landing.html)
│   ├── getting-started.md
│   ├── concepts/
│   │   ├── abstraction.md
│   │   ├── tiles-and-atoms.md
│   │   └── schedules.md
│   └── examples/
│       └── spmv.md
└── _site/                    # Build output (gitignored)
```

### 10.3 What gets committed vs. generated

**Committed** (tracked in git):
- `site/` directory (templates, CSS, JS, fonts, content, build script,
  Doxyfile).
- `DESIGN.md` (this document).

**Generated** (by CI, never committed):
- `build/doxygen-xml/` — doxygen XML output.
- `_site/` — final static site.

---

## 11. Accessibility

- **Color contrast**: all text/background pairs meet WCAG AA minimum.
  Primary text exceeds AAA.
- **Focus indicators**: all interactive elements have a visible
  `outline: 2px solid var(--accent)` with `outline-offset: 2px` on
  `:focus-visible`. No `outline: none` anywhere.
- **Skip link**: hidden "Skip to content" link appears on Tab, jumps
  past navigation.
- **Semantic HTML**: `<nav>`, `<main>`, `<article>`, `<aside>` (TOC),
  `<footer>`. Headings follow a strict h1→h2→h3 hierarchy per page.
- **Reduced motion**: `@media (prefers-reduced-motion: reduce)` disables
  the hero canvas animation and all transition durations.
- **Code blocks**: wrapped in `<pre><code>` with `role="code"`. Language
  identified via `data-language` attribute.
- **Images/diagrams**: all SVG diagrams include `<title>` and `<desc>`
  elements with meaningful descriptions.

---

## 12. Performance Budget

```
Metric              Target        Rationale
─────────────────────────────────────────────────────────────
First paint         < 800ms       No JS blocks rendering
LCP                 < 1.5s        Hero text renders server-side; canvas lazy
CLS                 < 0.05        Fonts preloaded, dimensions explicit
Total page weight   < 300KB       No frameworks; fonts subset to Latin
JS bundle           < 40KB        hero-canvas + sidebar + copy-code + toc
CSS total           < 20KB        Hand-written, no utility framework
Font files          < 150KB       3 fonts × 2 weights, woff2 only, subset
```

Fonts are preloaded with `<link rel="preload" as="font" crossorigin>`.
The hero canvas JS is loaded with `defer` and only initializes after
`DOMContentLoaded`. Critical CSS (above-the-fold) is inlined in
`<head>`.

---

## 13. Browser Support

Target: last 2 versions of Chrome, Firefox, Safari, Edge. No IE.

CSS features used that require this baseline:
- `backdrop-filter` (Safari 9+, Chrome 76+)
- CSS custom properties (all modern)
- `gap` in flexbox (Chrome 84+, Firefox 63+, Safari 14.1+)
- `font-variant-ligatures` (all modern)
- `IntersectionObserver` (all modern, polyfill not needed)

---

## 14. Design Tokens Summary (CSS Custom Properties)

All tokens defined in `site/static/css/tokens.css` as a single source
of truth. Components reference tokens, never raw values.

```css
:root {
  /* Colors — §2 */
  --bg-primary: #07070D;
  --bg-elevated: #0F0F1A;
  --bg-surface: #161625;
  --bg-subtle: #1E1E32;

  --text-primary: #E8E8F0;
  --text-secondary: #9090A8;
  --text-tertiary: #5A5A72;

  --accent: #6366F1;
  --accent-hover: #818CF8;
  --accent-muted: rgba(99, 102, 241, 0.2);
  --accent-glow: rgba(99, 102, 241, 0.25);

  --border: #1E1E32;
  --border-accent: rgba(99, 102, 241, 0.4);

  --success: #10B981;
  --warning: #F59E0B;
  --error: #EF4444;

  /* Typography — §3 */
  --font-code: 'JetBrains Mono', 'Fira Code', monospace;
  --font-heading: 'Space Grotesk', 'Inter', sans-serif;
  --font-body: 'Inter', system-ui, sans-serif;

  --text-xs: 0.8rem;
  --text-sm: 0.875rem;
  --text-base: 1rem;
  --text-lg: 1.25rem;
  --text-xl: 1.5625rem;
  --text-2xl: 1.953rem;
  --text-3xl: 2.441rem;
  --text-4xl: 3.052rem;

  /* Spacing — §4.3 */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 24px;
  --space-6: 32px;
  --space-7: 48px;
  --space-8: 64px;
  --space-9: 96px;

  /* Motion — §8.4 */
  --duration-fast: 100ms;
  --duration-normal: 200ms;
  --duration-slow: 400ms;
  --ease-default: cubic-bezier(0.4, 0, 0.2, 1);
  --ease-out: cubic-bezier(0, 0, 0.2, 1);
  --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);

  /* Layout — §4.1 */
  --sidebar-width: 240px;
  --toc-width: 200px;
  --content-max: 720px;
  --page-max: 1200px;
}
```

---

## 15. Implementation Order

Phase 1 — **Foundation** (scaffold + tokens + landing page):
1. Create `site/` directory structure per §10.2.
2. Implement CSS tokens (`tokens.css`).
3. Build `base.html` template with nav, sidebar shell, footer.
4. Build landing page (`landing.html` + `hero-canvas.js`).
5. Deploy a static "coming soon" version via GitHub Pages.

Phase 2 — **Documentation pages**:
6. Write `build.py` markdown → HTML pipeline.
7. Port existing `docs/*.md` content into the site.
8. Write new conceptual docs (tiles & atoms, schedules overview).
9. Implement sidebar navigation with scroll spy.

Phase 3 — **API reference**:
10. Configure Doxyfile for XML output.
11. Write `doxygen_to_html.py` to transform XML → themed HTML.
12. Generate API cards for all public symbols.
13. Cross-link API pages ↔ documentation pages.

Phase 4 — **Polish**:
14. Interactive schedule diagrams (animated SVG).
15. Performance audit against §12 budget.
16. Accessibility audit (keyboard nav, screen reader testing).
17. Open Graph image, favicon, meta tags.
18. Final review of all responsive breakpoints.
