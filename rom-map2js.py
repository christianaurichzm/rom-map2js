import argparse
from collections import Counter
import struct
from pathlib import Path
from PIL import Image, ImageChops, ImageStat
from tqdm import tqdm


TILE_SIZE = 16
TARGET_W, TARGET_H = 20, 14


MAP_HEADER_LAYOUT_PTR = 0x00
MAP_HEADER_LABEL_ID = 0x15
LAYOUT_WIDTH_OFFSET = 0x00
LAYOUT_HEIGHT_OFFSET = 0x04
LAYOUT_BLOCKS_PTR_OFFSET = 0x0C
LAYOUT_GLOBAL_TILESET_PTR = 0x10
LAYOUT_LOCAL_TILESET_PTR = 0x14


FIRERED_CHAR_TABLE = {
    0x00: ' ', 0x01: 'À', 0x02: 'Á', 0x03: 'Â', 0x04: 'Ç', 0x05: 'È', 0x06: 'É', 0x07: 'Ê',
    0x08: 'Ë', 0x09: 'Ì', 0x0B: 'Î', 0x0C: 'Ï', 0x0D: 'Ò', 0x0E: 'Ó', 0x0F: 'Ô', 0x10: 'Œ',
    0x11: 'Ù', 0x12: 'Ú', 0x13: 'Û', 0x14: 'Ñ', 0x15: 'ß', 0x16: 'à', 0x17: 'á', 0x19: 'ç',
    0x1A: 'è', 0x1B: 'é', 0x1C: 'ê', 0x1D: 'ë', 0x1E: 'ì', 0x20: 'î', 0x21: 'ï', 0x22: 'ò',
    0x23: 'ó', 0x24: 'ô', 0x25: 'œ', 0x26: 'ù', 0x27: 'ú', 0x28: 'û', 0x29: 'ñ', 0x2A: 'º',
    0x2B: 'ª', 0x2D: '&', 0x2E: '+', 0x34: '[Lv]', 0x35: '=', 0x36: ';', 0x51: '¿', 0x52: '¡',
    0x53: '[pk]', 0x54: '[mn]', 0x55: '[po]', 0x56: '[ké]', 0x57: '[bl]', 0x58: '[oc]',
    0x59: '[k]', 0x5A: 'Í', 0x5B: '%', 0x5C: '(', 0x5D: ')', 0x68: 'â', 0x6F: 'í', 0x79: '[U]',
    0x7A: '[D]', 0x7B: '[L]', 0x7C: '[R]', 0x85: '<', 0x86: '>', 0xA1: '0', 0xA2: '1',
    0xA3: '2', 0xA4: '3', 0xA5: '4', 0xA6: '5', 0xA7: '6', 0xA8: '7', 0xA9: '8', 0xAA: '9',
    0xAB: '!', 0xAC: '?', 0xAD: '.', 0xAE: '-', 0xAF: '·', 0xB0: '...', 0xB1: '«', 0xB2: '»',
    0xB3: '\'', 0xB4: '\'', 0xB5: '|m|', 0xB6: '|f|', 0xB7: '$', 0xB8: ',', 0xB9: '*',
    0xBA: '/', 0xBB: 'A', 0xBC: 'B', 0xBD: 'C', 0xBE: 'D', 0xBF: 'E', 0xC0: 'F', 0xC1: 'G',
    0xC2: 'H', 0xC3: 'I', 0xC4: 'J', 0xC5: 'K', 0xC6: 'L', 0xC7: 'M', 0xC8: 'N', 0xC9: 'O',
    0xCA: 'P', 0xCB: 'Q', 0xCC: 'R', 0xCD: 'S', 0xCE: 'T', 0xCF: 'U', 0xD0: 'V', 0xD1: 'W',
    0xD2: 'X', 0xD3: 'Y', 0xD4: 'Z', 0xD5: 'a', 0xD6: 'b', 0xD7: 'c', 0xD8: 'd', 0xD9: 'e',
    0xDA: 'f', 0xDB: 'g', 0xDC: 'h', 0xDD: 'i', 0xDE: 'j', 0xDF: 'k', 0xE0: 'l', 0xE1: 'm',
    0xE2: 'n', 0xE3: 'o', 0xE4: 'p', 0xE5: 'q', 0xE6: 'r', 0xE7: 's', 0xE8: 't', 0xE9: 'u',
    0xEA: 'v', 0xEB: 'w', 0xEC: 'x', 0xED: 'y', 0xEE: 'z', 0xEF: '|>|', 0xF0: ':', 0xF1: 'Ä',
    0xF2: 'Ö', 0xF3: 'Ü', 0xF4: 'ä', 0xF5: 'ö', 0xF6: 'ü', 0xF7: '|A|', 0xF8: '|V|',
    0xF9: '|<|', 0xFA: '|nb|', 0xFB: '|nb2|', 0xFC: '|FC|', 0xFD: '|FD|', 0xFE: '|br|',
}


_blk_cache = {}


class DecompressionError(ValueError):
    pass


def u32(b, o):
    return struct.unpack("<I", b[o:o+4])[0]


def u16(b, o):
    return struct.unpack("<H", b[o:o+2])[0]


def ptr(b, o):
    return u32(b, o) - 0x08000000


def is_ptr(b, o):
    return ptr(b, o) > 0


def bits(byte):
    return ((byte >> 7) & 1,
            (byte >> 6) & 1,
            (byte >> 5) & 1,
            (byte >> 4) & 1,
            (byte >> 3) & 1,
            (byte >> 2) & 1,
            (byte >> 1) & 1,
            (byte) & 1)


def decompress_raw_lzss10(indata, decompressed_size, _overlay=False):
    data = bytearray()

    it = iter(indata)

    if _overlay:
        disp_extra = 3
    else:
        disp_extra = 1

    def writebyte(b):
        data.append(b)
    def readbyte():
        return next(it)
    def readshort():
        a = next(it)
        b = next(it)
        return (a << 8) | b
    def copybyte():
        data.append(next(it))

    while len(data) < decompressed_size:
        b = readbyte()
        flags = bits(b)
        for flag in flags:
            if flag == 0:
                copybyte()
            elif flag == 1:
                sh = readshort()
                count = (sh >> 0xc) + 3
                disp = (sh & 0xfff) + disp_extra

                for _ in range(count):
                    writebyte(data[-disp])
            else:
                raise ValueError(flag)

            if decompressed_size <= len(data):
                break

    if len(data) != decompressed_size:
        raise DecompressionError("Decompressed size does not match the expected size")

    return data

def decompress_raw_lzss11(indata, decompressed_size):
    data = bytearray()

    it = iter(indata)

    def writebyte(b):
        data.append(b)
    def readbyte():
        return next(it)
    def copybyte():
        data.append(next(it))

    while len(data) < decompressed_size:
        b = readbyte()
        flags = bits(b)
        for flag in flags:
            if flag == 0:
                copybyte()
            elif flag == 1:
                b = readbyte()
                indicator = b >> 4

                if indicator == 0:
                    count = (b << 4)
                    b = readbyte()
                    count += b >> 4
                    count += 0x11
                elif indicator == 1:
                    count = ((b & 0xf) << 12) + (readbyte() << 4)
                    b = readbyte()
                    count += b >> 4
                    count += 0x111
                else:
                    count = indicator
                    count += 1

                disp = ((b & 0xf) << 8) + readbyte()
                disp += 1

                try:
                    for _ in range(count):
                        writebyte(data[-disp])
                except IndexError:
                    raise Exception(count, disp, len(data), sum(1 for x in it) )
            else:
                raise ValueError(flag)

            if decompressed_size <= len(data):
                break

    if len(data) != decompressed_size:
        raise DecompressionError("Decompressed size does not match the expected size")

    return data


def decompress_bytes(data):
    header = data[:4]
    if header[0] == 0x10:
        decompress_raw = decompress_raw_lzss10
    elif header[0] == 0x11:
        decompress_raw = decompress_raw_lzss11
    else:
        raise DecompressionError("Not as lzss-compressed file")

    decompressed_size, = struct.unpack("<L", header[1:] + b'\x00')

    data = data[4:]
    return decompress_raw(data, decompressed_size)


def load_rom(path):
    return Path(path).read_bytes()


def load_strings(buffer, offset_hex="0x3eecfc"):
    off = int(offset_hex, 16)
    out, s = [], ""
    while True:
        c = buffer[off]
        off += 1
        if c == 0xFF:
            out.append(s)
            s = ""
            continue
        if c not in FIRERED_CHAR_TABLE:
            break
        s += FIRERED_CHAR_TABLE[c]
    return out


def read_map_header(b, off):
    return ptr(b, off + MAP_HEADER_LAYOUT_PTR), b[off + MAP_HEADER_LABEL_ID]


def read_map_layout(b, lp):
    w = u32(b, lp + LAYOUT_WIDTH_OFFSET)
    h = u32(b, lp + LAYOUT_HEIGHT_OFFSET)
    return (
        w,
        h,
        ptr(b, lp + LAYOUT_BLOCKS_PTR_OFFSET),
        ptr(b, lp + LAYOUT_GLOBAL_TILESET_PTR),
        ptr(b, lp + LAYOUT_LOCAL_TILESET_PTR),
    )


def read_map_blocks(b, bp, w, h):
    spr = {}
    idx = 0
    for y in range(h):
        for x in range(w):
            spr[(x, y)] = u16(b, bp + idx * 2) & 0x3FF
            idx += 1
    return spr


def read_block(b, off, i):
    blk = []
    for j in range(8):
        d = u16(b, off + i * 16 + j * 2)
        blk.append(((d >> 12) & 0xF, d & 0x3FF, (d >> 10) & 3))
    return blk


def read_tileset(b, ts):
    if ts <= 0:
        return [], [], []
    f, p = struct.unpack("<2B", b[ts:ts+2])
    ip = ptr(b, ts + 4)
    pp = ptr(b, ts + 8)
    pr = range(7) if p == 0 else range(7, 16)
    pals = []
    for i in pr:
        pal = []
        for j in range(16):
            c = u16(b, pp + i * 32 + j * 2)
            r, g, bl = c & 0x1F, (c >> 5) & 0x1F, c >> 10
            pal.append((r * 8, g * 8, bl * 8))
        pals.append(pal)
    tiles = []
    if f:
        try:
            raw = decompress_bytes(b[ip:])
        except Exception:
            raw = b""
        for i in range(0, len(raw), 32):
            t = []
            for j in range(64):
                px = raw[i + j // 2]
                px = (px & 0xF) if j % 2 == 0 else px >> 4
                t.append(px)
            tiles.append(t)
    bp, ep = ptr(b, ts + 12), ptr(b, ts + 20)
    blks = [read_block(b, bp, i) for i in range((ep - bp) // 16)]
    return pals, tiles, blks


def build_tileset_table(img_path, tileset_bg):
    img = Image.open(img_path).convert("RGBA")
    w, h = img.size
    tiles = []
    h2i = {}
    for ty in range(0, h, TILE_SIZE * 4):
        for tx in range(0, w, TILE_SIZE * 4):
            blk = img.crop((tx, ty, tx + TILE_SIZE * 4, ty + TILE_SIZE * 4))
            if blk.getpixel((0, 0))[:3] == (255, 0, 0):
                blk = blk.crop((1, 1, blk.width - 1, blk.height - 1))
            t = blk.resize((TILE_SIZE, TILE_SIZE), Image.NEAREST)
            data = [(0, 0, 0, 0) if p[:3] == tileset_bg else p for p in t.getdata()]
            t.putdata(data)
            t = normalize_gba(t)
            tiles.append(t)
            h2i[hash(t.tobytes())] = len(tiles) - 1
    return tiles, h2i


def detect_tileset_bg(img_path):
    img = Image.open(img_path).convert("RGBA")
    pixels = list(img.getdata())
    pixels_rgb = [p[:3] for p in pixels]
    filtered = [c for c in pixels_rgb if c != (255, 0, 0)]
    most_common = Counter(filtered).most_common(1)[0][0]
    return most_common


def normalize_gba(img):
    d = [((r >> 3) << 3, (g >> 3) << 3, (b >> 3) << 3, a) for r, g, b, a in img.getdata()]
    o = Image.new("RGBA", img.size)
    o.putdata(d)
    return o


def tile_distance(a, b):
    diff = ImageChops.difference(a, b)
    return sum(ImageStat.Stat(diff).sum) / 261120


def mask_distance(a, b):
    ma = [1 if p[3] else 0 for p in a.getdata()]
    mb = [1 if p[3] else 0 for p in b.getdata()]
    return sum(1 for x, y in zip(ma, mb) if x != y) / len(ma)


def render_block(pals, tiles, blks, idx):
    img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE))
    blank = [0] * 64
    for i, (pal, tid, attr) in enumerate(blks[idx]):
        ox, oy = (i % 2) * 8, (i // 2 % 2) * 8
        xf, yf = attr & 1, attr & 2
        tile = tiles[tid] if tid < len(tiles) else blank
        for pi, px in enumerate(tile):
            if px == 0:
                continue
            cx = 7 - (pi % 8) if xf else pi % 8
            cy = 7 - (pi // 8) if yf else pi // 8
            img.putpixel((ox + cx, oy + cy), pals[pal][px] + (255,))
    return normalize_gba(img)


def block_to_tileindex(pals, tiles, blks, num, all_tiles, h2i, blank_idx):
    key = (id(pals), id(tiles), id(blks), num)
    if key in _blk_cache:
        return _blk_cache[key]
    img = render_block(pals, tiles, blks, num)
    img = normalize_gba(img)
    h = hash(img.tobytes())
    if h in h2i:
        idx = h2i[h]
        _blk_cache[key] = idx
        return idx
    best, bd, md = None, 1, 1
    for i, t in enumerate(all_tiles):
        pd = tile_distance(img, t)
        md2 = mask_distance(img, t)
        if md2 < .05 and pd < bd:
            best, bd, md = i, pd, md2
        elif md2 < md:
            best, bd, md = i, pd, md2
    if md < .05 and bd < .10:
        idx = best
    else:
        idx = blank_idx if md < .05 else len(all_tiles)
        if idx == len(all_tiles):
            all_tiles.append(img)
    h2i[h] = idx
    _blk_cache[key] = idx
    return idx


def find_horizontal_gap(full, blank_idx, min_gap=1, occ_thr=0.20):
    h = len(full)
    w = len(full[0])
    occ = [sum(1 for t in row if t != blank_idx) / w for row in full]
    gaps = []
    s = None
    for y, o in enumerate(occ):
        if o <= occ_thr:
            s = s if s is not None else y
        elif s is not None:
            gaps.append((s, y - 1))
            s = None
    if s is not None:
        gaps.append((s, h - 1))
    best = None
    for g0, g1 in gaps:
        L = g1 - g0 + 1
        if L >= min_gap and g0 >= TARGET_H // 2 and (h - 1 - g1) >= TARGET_H // 2:
            if best is None or L > (best[1] - best[0] + 1):
                best = (g0, g1)
    return best


def map_to_obj(b, header, strings, h2i, tiles, fallback, blank_idx, tq=None):
    lp, label = read_map_header(b, header)
    w, h, bp, gts, lts = read_map_layout(b, lp)
    name = strings[label - 88] if 0 <= label - 88 < len(strings) else fallback
    spr = read_map_blocks(b, bp, w, h)
    pals, tset, blks = read_tileset(b, gts)
    ep, et, eb = read_tileset(b, lts)
    pals += ep
    tset += et
    blks += eb

    full = [[None] * w for _ in range(h)]
    total = w * h
    if tq:
        tq.reset(total=total)
    cnt = 0
    for y in range(h):
        for x in range(w):
            full[y][x] = block_to_tileindex(
                pals, tset, blks, spr[(x, y)], tiles, h2i, blank_idx
            )
            cnt += 1
            if tq:
                tq.update(1)

    while len(full[0]) > TARGET_W and all(r[0] == blank_idx for r in full):
        full = [r[1:] for r in full]
    while len(full[0]) > TARGET_W and all(r[-1] == blank_idx for r in full):
        full = [r[:-1] for r in full]
    while len(full) > 2 * TARGET_H and all(t == blank_idx for t in full[0]):
        full.pop(0)
    while len(full) > 2 * TARGET_H and all(t == blank_idx for t in full[-1]):
        full.pop()

    gap = find_horizontal_gap(full, blank_idx)
    if gap:
        parts = [(0, gap[0]), (gap[1] + 1, len(full))]
    else:
        if h <= TARGET_H:
            parts = [(0, h)]
        else:
            parts = [(0, h // 2), (h // 2, h)]

    pages = []
    for top, bottom in parts:
        if bottom <= top:
            continue
        rows = []
        for y in range(top, bottom):
            line = (full[y] if y < len(full) else [blank_idx] * w)[:TARGET_W]
            line += [blank_idx] * (TARGET_W - len(line))
            rows.append([{"tileIndex": t, "blocked": False, "warp": "undefined"} for t in line])
        while len(rows) < TARGET_H:
            rows.append([{"tileIndex": blank_idx, "blocked": False, "warp": "undefined"}] * TARGET_W)
        p_name = f"{name}_p{len(pages)}"
        pages.append((p_name, rows))
    return pages


def js_grid(rows):
    L = []
    for row in rows:
        L.append("  [" + ", ".join(
            f"{{tileIndex: {c['tileIndex']}, blocked: {str(c['blocked']).lower()}, warp: undefined}}" for c in row
        ) + "]")
    return "[\n" + ",\n".join(L) + "\n]"


def sanitize_name(name):
    import re
    return re.sub(r'\W|^(?=\d)', '_', name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rom")
    parser.add_argument("--tileset", required=True)
    parser.add_argument("--bank", type=int)
    parser.add_argument("--map", type=int)
    parser.add_argument("-o", "--output", default="maps")
    args = parser.parse_args()

    tileset_bg = detect_tileset_bg(args.tileset)
    tiles, h2i = build_tileset_table(Path(args.tileset), tileset_bg)

    blank_idx = next((i for i, t in enumerate(tiles) if all(px[3] == 0 for px in t.getdata())), None)
    if blank_idx is None:
        raise RuntimeError("No transparent tile")

    data = load_rom(args.rom)
    strings = load_strings(data)

    bto = 0x3526A8
    off = bto
    banks = []
    ptrs = []
    while is_ptr(data, off):
        ptrs.append(ptr(data, off))
        off += 4

    for i, bp in enumerate(ptrs):
        if i == 42:
            break
        m = []
        o = bp
        nxt = ptrs[i+1] if i+1 < len(ptrs) else 0
        while is_ptr(data, o):
            m.append(ptr(data, o))
            o += 4
            if o == nxt:
                break
        banks.append(m)

    out = Path(args.output)
    out.mkdir(exist_ok=True)

    page_refs = []
    for bi, bank in enumerate(banks):
        if args.bank is not None and bi != args.bank:
            continue
        for mi, header in enumerate(bank):
            if args.map is not None and mi != args.map:
                continue
            fallback = f"bank{bi}_map{mi}"
            lp, label_id = read_map_header(data, header)
            if 0 <= label_id - 88 < len(strings):
                name = strings[label_id - 88]
            elif mi < len(strings):
                name = strings[mi]
            else:
                name = fallback

            if not name or len(name.strip()) < 2:
                name = fallback

            w, h, _, _, _ = read_map_layout(data, lp)
            gap = find_horizontal_gap([[0] * w for _ in range(h)], blank_idx)
            if gap:
                parts = [(0, gap[0]), (gap[1] + 1, h)]
            else:
                parts = [(0, h)] if h <= TARGET_H else [(0, h // 2), (h // 2, h)]
            for pi in range(len(parts)):
                page_refs.append((bi, mi, header, name, pi))

    gbar = tqdm(total=len(page_refs), desc="All pages", ncols=110)
    for bi, mi, header, name, pi in page_refs:
        lp, _ = read_map_header(data, header)
        w, h, _, _, _ = read_map_layout(data, lp)
        total = w * h
        with tqdm(total=total, desc=f"Page {name}_p{pi}", leave=False, ncols=110) as pbar:
            pages = map_to_obj(data, header, strings, h2i, tiles, name, blank_idx, tq=pbar)
        page_name, rows = pages[pi]
        array = js_grid(rows)
        var = sanitize_name(page_name)
        with open(out / f"{page_name}.js", "w", encoding="utf8") as fp:
            fp.write(f"export const {var} = {array}\n")
        gbar.update(1)
    gbar.close()
    print(f"\n{len(page_refs)} tile arrays exported as JS in {out}/")


if __name__ == "__main__":
    main()
  
