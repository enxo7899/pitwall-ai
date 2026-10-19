// Page design of the 2-page report (docs/REPORT.pdf), filled by scripts/tools/build_report.py.
// A white page, dark text and a single accent colour (links and headings); tables in the
// "booktabs" style: a rule above, below and under the header, no grid lines.

#let ink = rgb("#1b1f24")
#let muted = rgb("#5b6470")
#let accent = rgb("#174a87")
#let hairline = rgb("#d5d9de")

#let report(title: "", subtitle: "", links: (), body) = {
  set document(title: title + ": " + subtitle, date: none) // no build timestamp, so rebuilds are byte-identical
  set page(
    paper: "a4",
    margin: (x: 2cm, top: 1.7cm, bottom: 1.6cm),
    footer: context {
      set text(size: 7.5pt, fill: muted)
      grid(
        columns: (1fr, auto),
        [#title · #subtitle], [#counter(page).display("1 / 1", both: true)],
      )
    },
  )
  set text(font: "Helvetica Neue", size: 9.4pt, fill: ink, lang: "en", region: "gb")
  set par(leading: 0.56em, spacing: 0.8em, justify: false)
  set strong(delta: 200)
  show link: set text(fill: accent)

  // Section headings: one size, accent colour, generous space above.
  show heading: it => block(above: 1.1em, below: 0.55em, sticky: true)[
    #text(size: 10.8pt, weight: "bold", fill: accent)[#it.body]
  ]

  // Lists: compact, with a small muted bullet.
  set list(marker: text(fill: muted)[•], indent: 0.15em, body-indent: 0.55em, spacing: 0.6em)

  // Tables: booktabs rules, comfortable padding, slightly smaller text.
  show figure.where(kind: table): set figure(gap: 0pt)
  show figure.where(kind: table): set block(above: 0.9em, below: 1em)
  show table: set text(size: 9pt)
  set table(stroke: (x, y) => if y > 0 { (top: 0.4pt + hairline) }, inset: (x: 0.45em, y: 0.48em))
  set table.hline(stroke: 0.6pt + ink)
  // Pandoc centres tables and their cells; text cells read better left-aligned.
  show table.cell: set align(left + top)
  show table: it => {
    let columns = if type(it.columns) == int { it.columns } else { it.columns.len() }
    let ruled = block.with(stroke: (top: 0.9pt + ink, bottom: 0.9pt + ink))
    if columns == 5 {
      // the results table: numbers centred under their headers, header row bold
      show table.cell: c => if c.x == 0 { c } else { align(center, c) }
      show table.cell.where(y: 0): set text(weight: "bold")
      ruled(it)
    } else {
      ruled(it)
    }
  }

  // Figures: caption in small muted text, without a "Figure 1:" prefix.
  show figure.where(kind: image): set block(above: 1em, below: 0.9em)
  show figure.caption: it => align(left, text(size: 8pt, fill: muted)[#it.body])

  // Title block.
  block(below: 0.9em)[
    #text(size: 22pt, weight: "bold", tracking: -0.015em)[#title]
    #v(0.55em, weak: true)
    #text(size: 12pt, fill: muted)[#subtitle]
  ]
  grid(
    columns: links.len(),
    column-gutter: 2.2em,
    ..links.map(((label, shown, url)) => [
      #text(size: 7pt, weight: "bold", fill: muted, tracking: 0.08em)[#upper(label)] \
      #link(url)[#text(size: 9pt)[#shown]]
    ]),
  )
  v(0.7em)
  line(length: 100%, stroke: 0.6pt + hairline)
  v(0.2em)

  body
}
