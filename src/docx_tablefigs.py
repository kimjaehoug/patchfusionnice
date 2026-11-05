"""Replace Tables 2, 3, 5, 6 and 7 of the ESWA manuscript by Figs. 3, 4, 6 and 7, add Fig. 5 after the main results
table, renumber the remaining tables (4, 8, 9 -> 2, 3, 4) and figures (3, 4 -> 8, 9) and their in-text citations, and
move the replaced tables verbatim to a new Supplementary section S23 (Tables S15-S19).

usage: docx_tablefigs.py MAIN_IN MAIN_OUT [SUPP_IN SUPP_OUT] --figs DIR
MAIN may contain the supplement (full archive version); otherwise SUPP_IN/SUPP_OUT receive section S23."""
import copy
import re
import sys
import zipfile
from lxml import etree
from PIL import Image

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
W14 = "http://schemas.microsoft.com/office/word/2010/wordml"
WP14 = "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"
q = lambda ns, t: f"{{{ns}}}{t}"

CAP = {
    3: "Fig. 3. Label-free observation profiles at three scales, computed from routine records without outcome labels. "
       "(a) Site: number of the 22 input variables never recorded. (b) Stay: share of 48-h windows with the laboratory "
       "variable or with the charting stream. (c) Patch: cadence within those windows, as the share of 3-h blocks with a "
       "laboratory test and of monitored hours with a charting or SpO2 record. Hatched cells: never recorded at the "
       "hospital. Values are listed in Supplementary Table S15.",
    4: "Fig. 4. Component damage, one component at a time: change in discrimination of the source-only backbone (hidden "
       "size 256) on 30,000 held-out development samples when one component of the observation profile is set to the "
       "target value. Points: mean over four seeds; bars: range over seeds. Colour marks the target whose value is "
       "applied; components are grouped by scale. Values are listed in Supplementary Table S16.",
    5: "Fig. 5. Change over the source-only backbone with full-data training (five-seed ensembles; values in Table 2) in "
       "AUROC (a) and AUPRC (b) at all eICU-CRD hospitals and at the four Hallym hospitals, an independent evaluation of "
       "the frozen design. Importance reweighting and generic counterfactual training are comparators; rate-level "
       "imagination, the proposed model without the patch scale and the proposed three-scale model are variants of the "
       "proposed imagination. Numbers give the change for the proposed model.",
    6: "Fig. 6. Contribution of the patch-scale steps. (a, b) Difference in AUROC and AUPRC of the proposed method with "
       "versus without the patch-scale steps (hidden size 256, 30% of development patients, four seeds; 95% cluster "
       "bootstrap CI, hospital clusters at low-testing eICU-CRD hospitals and patient clusters at the Hallym hospitals). "
       "(c) O/E without (open) and with (filled) the steps; the dotted line marks O/E = 1. (d) 24-h AUROC on held-out "
       "development patients before (open) and after (filled) the charting cadence of the Glasgow Coma Scale, urine "
       "output and norepinephrine was lowered to the low-testing eICU-CRD rate, keeping every stay's streams (hidden "
       "size 256, four-seed ensembles); the columns give the loss in 24-h and in-hospital AUROC. Values are listed in "
       "Supplementary Tables S17 and S18.",
    7: "Fig. 7. Source–target trade-off for death within 24 h (five-seed ensembles): change relative to the source-only "
       "backbone in AUROC (a) and AUPRC (b) on the held-out MIMIC-IV development fold and at the two targets. Bars: 95% "
       "patient-cluster bootstrap CI of the development difference. Values are listed in Supplementary Table S19.",
}
# replaced table -> (figure placed there or None, supplementary table number, figure it documents)
MOVE = {2: (3, 15, "Fig. 3"), 3: (4, 16, "Fig. 4"), 5: (6, 17, "Fig. 6a–c"), 6: (None, 18, "Fig. 6d"), 7: (7, 19, "Fig. 7")}
SPECIFIC = {"better at both external sources up to 48 h (Table 4)": "better at both external sources up to 48 h (Table 2; Fig. 5)",
            "(Table 5; ": "(Fig. 6a–c; ", "(Table 6)": "(Fig. 6d)", "(Table 7)": "(Fig. 7)"}
GLOBAL = {"Table 2": "Fig. 3", "Table 3": "Fig. 4", "Table 4": "Table 2", "Table 8": "Table 3", "Table 9": "Table 4",
          "Fig. 3": "Fig. 8", "Fig. 4": "Fig. 9"}


def ptext(p):
    return "".join(t.text or "" for t in p.iter(q(W, "t")))


def sub_in_par(p, pattern, repl):
    """regex substitution on the paragraph's text across w:t nodes (one pass); returns number of substitutions"""
    ts = list(p.iter(q(W, "t")))
    if not ts: return 0
    full = "".join(t.text or "" for t in ts)
    ms = list(re.finditer(pattern, full))
    for m in reversed(ms):
        new = repl(m) if callable(repl) else m.expand(repl)
        s, e = m.span(); pos = 0; first = True
        for t in ts:
            txt = t.text or ""; b0, b1 = pos, pos + len(txt); pos = b1
            if b1 <= s or b0 >= e: continue
            a, b = max(s, b0) - b0, min(e, b1) - b0
            t.text = txt[:a] + (new if first else "") + txt[b:]; first = False
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return len(ms)


def strip_ids(el):
    for e in el.iter():
        for att in (q(W14, "paraId"), q(W14, "textId")):
            if att in e.attrib: del e.attrib[att]
    return el


def set_caption_text(p, text):
    """keep pPr and first run's rPr, replace the runs by one run with text"""
    rpr = None
    for r in p.findall(q(W, "r")):
        rp = r.find(q(W, "rPr"))
        if rp is not None: rpr = copy.deepcopy(rp)
        break
    for ch in list(p):
        if ch.tag != q(W, "pPr"): p.remove(ch)
    r = etree.SubElement(p, q(W, "r"))
    if rpr is not None: r.append(rpr)
    t = etree.SubElement(r, q(W, "t")); t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return p


class Doc:
    def __init__(self, path):
        self.zin = zipfile.ZipFile(path)
        self.parts = {n: self.zin.read(n) for n in self.zin.namelist()}
        self.root = etree.fromstring(self.parts["word/document.xml"])
        self.body = self.root.find(q(W, "body"))
        self.rels = etree.fromstring(self.parts["word/_rels/document.xml.rels"])
        self.new_media = {}

    def add_image(self, png):
        ids = [int(re.sub(r"\D", "", r.get("Id")) or 0) for r in self.rels]
        rid = f"rId{max(ids) + 1}"
        n = 1
        while f"word/media/image{n}.png" in self.parts or f"word/media/image{n}.png" in self.new_media: n += 1
        name = f"media/image{n}.png"
        self.new_media["word/" + name] = open(png, "rb").read()
        etree.SubElement(self.rels, "Relationship", Id=rid,
                         Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", Target=name)
        ct = self.parts["[Content_Types].xml"]
        if b'Extension="png"' not in ct:
            self.parts["[Content_Types].xml"] = ct.replace(b"<Default ", b'<Default Extension="png" ContentType="image/png"/><Default ', 1)
        return rid

    def save(self, path):
        self.parts["word/document.xml"] = etree.tostring(self.root, xml_declaration=True, encoding="UTF-8", standalone=True)
        self.parts["word/_rels/document.xml.rels"] = etree.tostring(self.rels, xml_declaration=True, encoding="UTF-8", standalone=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for info in self.zin.infolist():
                z.writestr(copy.copy(info), self.parts[info.filename])
            for n, d in self.new_media.items():
                z.writestr(n, d)


def find_par(els, prefix):
    for i, e in enumerate(els):
        if e.tag == q(W, "p") and ptext(e).startswith(prefix): return i
    raise KeyError(prefix)


def main(argv):
    figs = argv[argv.index("--figs") + 1]
    args = [a for a in argv if not a.startswith("--") and a != figs]
    main_in, main_out = args[0], args[1]
    supp = args[2:4] if len(args) >= 4 else None
    d = Doc(main_in)
    els = list(d.body)
    supp_start = next((i for i, e in enumerate(els) if e.tag == q(W, "p") and ptext(e).strip() == "Supplementary material"), len(els))
    main_els = set(els[:supp_start])
    # template figure paragraph and caption: the current Fig. 3 (calibration)
    fi = find_par(els, "Fig. 3. Calibration")
    fig_tpl, cap_tpl = els[fi - 1], els[fi]
    assert fig_tpl.find(".//" + q(W, "drawing")) is not None
    docpr_ids = [int(e.get("id")) for e in d.root.iter(q(WP, "docPr"))]
    next_id = [max(docpr_ids) + 1]

    def figure_pars(n):
        rid = d.add_image(f"{figs}/Fig{n}_600dpi.png")
        w, h = Image.open(f"{figs}/Fig{n}_600dpi.png").size
        fp = copy.deepcopy(fig_tpl)
        for e in fp.iter(q(W, "lastRenderedPageBreak")): e.getparent().remove(e)
        cx = int(fp.find(".//" + q(WP, "extent")).get("cx")); cy = int(round(cx * h / w))
        for ext in list(fp.iter(q(WP, "extent"))) + list(fp.iter(q(A, "ext"))):
            if ext.get("cx") is not None: ext.set("cx", str(cx)); ext.set("cy", str(cy))
        for e in list(fp.iter(q(WP, "docPr"))) + list(fp.iter(q(PIC, "cNvPr"))):
            e.set("id", str(next_id[0])); e.set("name", f"Figure {n}")
        next_id[0] += 1
        fp.find(".//" + q(A, "blip")).set(q(R, "embed"), rid)
        for e in [fp] + list(fp.iter(q(WP, "inline"))):
            for att in (q(W14, "paraId"), q(W14, "textId"), q(WP14, "anchorId"), q(WP14, "editId")):
                if att in e.attrib: del e.attrib[att]
        cp = strip_ids(set_caption_text(copy.deepcopy(cap_tpl), CAP[n]))
        return strip_ids(fp), cp

    # 1. body citations (main text only, not captions); specific forms first, then a simultaneous global map
    for e in els[:supp_start]:
        if e.tag != q(W, "p"): continue
        s = ptext(e)
        if re.match(r"^(Table \d+\.|Fig\. \d+\.)", s): continue
        pat = "|".join(re.escape(k) for k in SPECIFIC) + r"|\b(?:Table [2-9]|Fig\. [34])(?!\d)"
        sub_in_par(e, pat, lambda m: SPECIFIC.get(m.group(0)) or GLOBAL.get(m.group(0), m.group(0)))
    # 2. captions of kept items
    for old, new in (("Table 4.", "Table 2."), ("Table 8.", "Table 3."), ("Table 9.", "Table 4."),
                     ("Fig. 3. Calibration", "Fig. 8. Calibration"), ("Fig. 4. Decision", "Fig. 9. Decision")):
        i = find_par(list(d.body), old)
        assert sub_in_par(list(d.body)[i], "^" + re.escape(old), new) == 1, old
    # 3. replace tables by figures; collect the moved tables
    moved = []
    for tno in (2, 3, 5, 6, 7):
        els = list(d.body)
        ci = find_par(els, f"Table {tno}. ")
        cap, tbl = els[ci], els[ci + 1]
        assert tbl.tag == q(W, "tbl"), tno
        fno, sno, ref = MOVE[tno]
        moved.append((sno, ref, re.sub(rf"^Table {tno}\. ", "", ptext(cap)), copy.deepcopy(cap), copy.deepcopy(tbl)))
        if fno is not None:
            fp, cp = figure_pars(fno)
            cap.addprevious(fp); cap.addprevious(cp)
        d.body.remove(cap); d.body.remove(tbl)
    # 4. Fig. 5 after the main results table (now Table 2)
    els = list(d.body)
    ti = find_par(els, "Table 2. Full-data")
    assert els[ti + 1].tag == q(W, "tbl")
    fp, cp = figure_pars(5)
    els[ti + 1].addnext(fp); fp.addnext(cp)
    # 5. supplementary section S23
    if supp:
        sd = Doc(supp[0]); target = sd
    else:
        sd = None; target = d
    tels = list(target.body)
    h_tpl = tels[find_par(tels, "S22. ")]
    cap_s = tels[find_par(tels, "Table S14. ")]
    txt_tpl = tels[find_par(tels, "S22. ") + 1]
    sect = tels[-1] if tels[-1].tag == q(W, "sectPr") else None
    anchor_add = (lambda x: sect.addprevious(x)) if sect is not None else (lambda x: target.body.append(x))
    h = strip_ids(set_caption_text(copy.deepcopy(h_tpl), "S23. Values of Figs. 3, 4, 6 and 7"))
    anchor_add(h)
    intro = strip_ids(set_caption_text(copy.deepcopy(txt_tpl), "Tables S15–S19 list the values plotted in Figs. 3, 4, 6 and 7 of "
                             "the main text, with the intervals and ranges that the figures show graphically."))
    anchor_add(intro)
    for sno, ref, rest, cap, tbl in moved:
        c = strip_ids(set_caption_text(copy.deepcopy(cap_s), f"Table S{sno}. {rest.rstrip()} Plotted in {ref}."))
        anchor_add(c); anchor_add(tbl)
    d.save(main_out)
    if sd is not None: sd.save(supp[1])
    print("done", main_out, supp[1] if supp else "")


if __name__ == "__main__":
    main(sys.argv[1:])
