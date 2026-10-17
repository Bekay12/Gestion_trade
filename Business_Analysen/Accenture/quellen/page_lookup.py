import re, sys

def build_index(path, banner_text):
    lines = open(path, encoding="utf-8").readlines()
    idx = []  # (line_no, page_label)
    for i, l in enumerate(lines):
        if banner_text in l:
            m = re.search(r'\|\s*([A-Za-z0-9-]+)\s*\|\s*$', l.strip())
            if m:
                idx.append((i, m.group(1)))
    return idx, lines

def page_for_line(idx, line_no):
    page = None
    for ln, p in idx:
        if ln <= line_no:
            page = p
        else:
            break
    return page

if __name__ == "__main__":
    path, banner, target_line = sys.argv[1], sys.argv[2], int(sys.argv[3])
    idx, _ = build_index(path, banner)
    print(page_for_line(idx, target_line))
