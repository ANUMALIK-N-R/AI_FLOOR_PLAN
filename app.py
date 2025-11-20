import streamlit as st
from io import BytesIO
import base64
import json
import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

# -------------------------
# (Embedded) GenerativeAdjacencyTetris (core logic - UNCHANGED)
# -------------------------
GRID_CELL_PX = 8
CANVAS_W_PX = 1200
CANVAS_H_PX = 800
WALL_THICKNESS_PX = 3

COLOR_MAP = {
    'Bedroom': [209, 224, 255], 'MasterBedroom': [173, 196, 255],
    'Bathroom': [200, 255, 255], 'Kitchen': [255, 248, 220],
    'DrawingRoom': [220, 255, 220], 'Dining': [255, 240, 245],
    'Lobby': [245, 235, 255], 'Balcony': [200, 255, 200],
    'Store': [220, 220, 220], 'Study': [255, 255, 200], 'Wall': [0,0,0]
}

ROOM_SPECS = {
    'MasterBedroom':{'pct': 0.15, 'aspect': (1.1, 1.5)},
    'Bedroom':{'pct': 0.13, 'aspect': (1.1, 1.5)},
    'DrawingRoom':{'pct': 0.18, 'aspect': (1.2, 1.8)},
    'Dining':{'pct': 0.12, 'aspect': (1.1, 1.6)},
    'Kitchen':{'pct': 0.10, 'aspect': (1.0, 2.0)},
    'Lobby': {'pct': 0.10, 'aspect': (1.0, 2.5)},
    'Bathroom':{'pct': 0.06, 'aspect': (1.0, 1.8)},
    'Balcony':{'pct': 0.05, 'aspect': (2.0, 4.0)},
    'Store':{'pct': 0.04, 'aspect': (1.0, 1.5)},
    'Study':{'pct': 0.04, 'aspect': (1.0, 1.5)}
}

PREFERRED_ADJ = {
    'Bedroom': ['Bathroom','Balcony','Lobby'],
    'MasterBedroom': ['Bathroom','Balcony','Lobby'],
    'Bathroom': ['Bedroom','MasterBedroom'],
    'Kitchen': ['Dining','Lobby'],
    'Dining': ['Kitchen','DrawingRoom'],
    'DrawingRoom': ['Lobby','Dining'],
    'Lobby': ['DrawingRoom','Kitchen','Bedroom'],
    'Balcony': ['DrawingRoom','MasterBedroom','Bedroom'],
    'Store': ['Kitchen'],
    'Study': ['DrawingRoom','Lobby']
}

FORBIDDEN_ADJ = {('Kitchen','Bathroom'), ('Bathroom','Kitchen')}

W_PREF_TOUCH = 300
W_TOUCH_ANY = 50
W_FORBID = -10000
W_EXTERIOR_BALCONY = 150
W_CENTER = 5
W_ALIGN = 10
MIN_W_CELLS = 4
MIN_H_CELLS = 3

class GenerativeParams:
    def __init__(self, seed=None):
        self.seed = seed if seed is not None else random.randint(0, 1000000)
        self.size_variation = random.uniform(0.85, 1.15)
        self.aspect_variation = random.uniform(0.9, 1.1)
        self.placement_strategy = random.choice(['center', 'corner_nw', 'corner_ne', 'corner_sw', 'corner_se', 'edge_n', 'edge_s', 'edge_e', 'edge_w'])
        self.priority_shuffle = random.random() < 0.3
        self.compact_strength = random.uniform(0.5, 1.5)
        self.rotation_preference = random.uniform(0.3, 0.7)
        self.adjacency_weight_mult = random.uniform(0.8, 1.2)
        self.exploration_factor = random.uniform(0.7, 1.3)
    def __repr__(self):
        return f"GenParams(seed={self.seed}, size_var={self.size_variation:.2f}, placement={self.placement_strategy})"

# small utilities
def px_to_cells(px, grid_cell=GRID_CELL_PX): return max(1, int(round(px / grid_cell)))
def cells_to_px(c, grid_cell=GRID_CELL_PX): return int(c * grid_cell)

def choose_font(size=14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()

class Room:
    def __init__(self, rtype, w_cells, h_cells, idx):
        self.type = rtype
        self.w = int(w_cells)
        self.h = int(h_cells)
        self.x = 0
        self.y = 0
        self.idx = idx
        self.placed = False
        self.doors = []
        self.color = COLOR_MAP.get(rtype, [200,200,200])
    def bbox(self):
        return (self.x, self.y, self.x + self.w, self.y + self.h)
    def center(self):
        return (self.x + self.w/2, self.y + self.h/2)
    def area_cells(self):
        return self.w * self.h
    def overlaps(self, other, buff=0):
        ax1, ay1, ax2, ay2 = self.bbox()
        bx1, by1, bx2, by2 = other.bbox()
        return not (ax2 + buff <= bx1 or bx2 + buff <= ax1 or ay2 + buff <= by1 or by2 + buff <= ay1)
    def touches(self, other):
        ax1, ay1, ax2, ay2 = self.bbox()
        bx1, by1, bx2, by2 = other.bbox()
        vertical_touch = (ax2 == bx1 or bx2 == ax1) and (min(ay2,by2) > max(ay1,by1))
        horizontal_touch = (ay2 == by1 or by2 == ay1) and (min(ax2,bx2) > max(ax1,bx1))
        return vertical_touch or horizontal_touch

class GenerativeAdjacencyTetris:
    def __init__(self, bhk=3, total_sqft=1500, canvas_w=CANVAS_W_PX, canvas_h=CANVAS_H_PX, grid_cell=GRID_CELL_PX, gen_params=None, verbose=False):
        self.bhk = bhk
        self.total_sqft = total_sqft
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.grid_cell = grid_cell
        self.grid_w = max(16, canvas_w // grid_cell)
        self.grid_h = max(12, canvas_h // grid_cell)
        self.verbose = verbose
        self.gen_params = gen_params if gen_params else GenerativeParams()
        random.seed(self.gen_params.seed)
        np.random.seed(self.gen_params.seed)
        self.rooms = []
        self.placed = []
        self.occupancy = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)
        self._prepare_rooms()

    def _prepare_rooms(self):
        template = ['DrawingRoom','Lobby','Dining','Kitchen','MasterBedroom']
        for _ in range(self.bhk - 1):
            template.append('Bedroom')
        for _ in range(self.bhk):
            template.append('Bathroom')
        template.append('Balcony')
        if self.total_sqft > 1800:
            template.append('Store')
            template.append('Study')
        total_pixels = self.canvas_w * self.canvas_h
        pixels_per_sqft = (total_pixels * 0.6) / max(1.0, self.total_sqft)
        counts = defaultdict(int)
        for r in template:
            counts[r] += 1
        idx = 0
        rooms = []
        for rtype, cnt in counts.items():
            spec = ROOM_SPECS.get(rtype, {'pct':0.08, 'aspect':(1.0,1.6)})
            for _ in range(cnt):
                per_room_sqft = (self.total_sqft * spec['pct']) / cnt
                per_room_sqft *= self.gen_params.size_variation
                per_room_px = per_room_sqft * pixels_per_sqft
                aspect_min, aspect_max = spec['aspect']
                aspect_range = aspect_max - aspect_min
                aspect = aspect_min + aspect_range * self.gen_params.aspect_variation
                aspect = random.uniform(max(0.1, aspect * 0.9), aspect * 1.1)
                h_px = math.sqrt(per_room_px / max(aspect, 1e-6))
                w_px = per_room_px / max(h_px, 1e-6)
                w_c = max(MIN_W_CELLS, px_to_cells(w_px, self.grid_cell))
                h_c = max(MIN_H_CELLS, px_to_cells(h_px, self.grid_cell))
                if random.random() < self.gen_params.rotation_preference:
                    w_c, h_c = h_c, w_c
                rooms.append(Room(rtype, w_c, h_c, idx))
                idx += 1
        priority_map = {'DrawingRoom':10,'Lobby':9,'Dining':8,'Kitchen':7,'MasterBedroom':6,'Bedroom':5,'Bathroom':4,'Balcony':3,'Store':2,'Study':2}
        if self.gen_params.priority_shuffle:
            priority_map = {k: v + random.uniform(-1, 1) for k, v in priority_map.items()}
        rooms.sort(key=lambda r: (priority_map.get(r.type,0), r.area_cells()), reverse=True)
        self.rooms = rooms
        if self.verbose:
            print(f"[INIT] {self.gen_params}")
            print(f"[INIT] Prepared {len(self.rooms)} rooms. Grid {self.grid_w}x{self.grid_h}")

    def _get_seed_position(self, room):
        strategy = self.gen_params.placement_strategy
        margin = 2
        if strategy == 'center':
            x = max(margin, (self.grid_w - room.w)//2)
            y = max(margin, (self.grid_h - room.h)//2)
        elif strategy == 'corner_nw':
            x = margin; y = margin
        elif strategy == 'corner_ne':
            x = max(margin, self.grid_w - room.w - margin); y = margin
        elif strategy == 'corner_sw':
            x = margin; y = max(margin, self.grid_h - room.h - margin)
        elif strategy == 'corner_se':
            x = max(margin, self.grid_w - room.w - margin); y = max(margin, self.grid_h - room.h - margin)
        elif strategy == 'edge_n':
            x = max(margin, (self.grid_w - room.w)//2); y = margin
        elif strategy == 'edge_s':
            x = max(margin, (self.grid_w - room.w)//2); y = max(margin, self.grid_h - room.h - margin)
        elif strategy == 'edge_e':
            x = max(margin, self.grid_w - room.w - margin); y = max(margin, (self.grid_h - room.h)//2)
        elif strategy == 'edge_w':
            x = margin; y = max(margin, (self.grid_h - room.h)//2)
        else:
            x = max(margin, (self.grid_w - room.w)//2); y = max(margin, (self.grid_h - room.h)//2)
        x += random.randint(-2, 2); y += random.randint(-2, 2)
        return x, y

    def _can_place(self, w,h,x,y):
        if x < 0 or y < 0 or x + w > self.grid_w or y + h > self.grid_h:
            return False
        sub = self.occupancy[y:y+h, x:x+w]
        return sub.sum() == 0

    def _place_room(self, room, x,y):
        room.x = int(x); room.y = int(y); room.placed = True
        self.occupancy[room.y:room.y+room.h, room.x:room.x+room.w] = 1
        self.placed.append(room)
        if self.verbose:
            print(f"[PLACE] {room.type} ({room.idx}) @ ({room.x},{room.y}) size {room.w}x{room.h}")

    def _touching_types_if_placed(self, w,h,x,y):
        touching = set()
        ax1, ay1, ax2, ay2 = x, y, x+w, y+h
        for other in self.placed:
            bx1, by1, bx2, by2 = other.bbox()
            vert = ((ax2 == bx1 or bx2 == ax1) and (min(ay2,by2) > max(ay1,by1)))
            horz = ((ay2 == by1 or by2 == ay1) and (min(ax2,bx2) > max(ax1,bx1)))
            if vert or horz:
                touching.add(other.type)
        return touching

    def _is_exterior(self, x,y,w,h):
        return (x == 0 or y == 0 or x + w == self.grid_w or y + h == self.grid_h)

    def _score_candidate(self, rtype, w,h,x,y):
        if not self._can_place(w,h,x,y):
            return -1e9
        touching = self._touching_types_if_placed(w,h,x,y)
        base = -5000 if (len(self.placed) > 0 and len(touching) == 0) else 0
        for t in touching:
            if (rtype, t) in FORBIDDEN_ADJ or (t, rtype) in FORBIDDEN_ADJ:
                return W_FORBID
        prefs = PREFERRED_ADJ.get(rtype, [])
        pref_hits = sum(1 for t in touching if t in prefs)
        score = base + pref_hits * W_PREF_TOUCH * self.gen_params.adjacency_weight_mult
        score += len(touching) * W_TOUCH_ANY
        if rtype == 'Balcony' and self._is_exterior(x,y,w,h):
            score += W_EXTERIOR_BALCONY
        if rtype == 'Bathroom' and self._is_exterior(x,y,w,h):
            score -= 150
        cx = x + w/2; cy = y + h/2
        if self.gen_params.placement_strategy.startswith('corner'):
            corner_dist = min(
                math.hypot(cx, cy),
                math.hypot(cx, self.grid_h - cy),
                math.hypot(self.grid_w - cx, cy),
                math.hypot(self.grid_w - cx, self.grid_h - cy)
            )
            score += -corner_dist * W_CENTER * 0.5
        elif self.gen_params.placement_strategy.startswith('edge'):
            edge_dist = min(cx, self.grid_w - cx, cy, self.grid_h - cy)
            score += -edge_dist * W_CENTER * 0.3
        else:
            center_dist = math.hypot(cx - self.grid_w/2, cy - self.grid_h/2)
            score += -center_dist * W_CENTER
        for other in self.placed:
            if abs(other.x - x) <= 1 or abs(other.y - y) <= 1:
                score += W_ALIGN
        return score

    def _generate_adjacent_candidates(self, room, max_candidates=1500):
        candidates = []
        if len(self.placed) == 0:
            x, y = self._get_seed_position(room)
            candidates.append((room.w, room.h, x, y, False))
            return candidates
        max_candidates = int(max_candidates * self.gen_params.exploration_factor)
        for other in self.placed:
            ox1, oy1, ox2, oy2 = other.bbox()
            x = ox1 - room.w
            for y in range(oy1 - room.h + 1, oy2):
                candidates.append((room.w, room.h, x, y, False))
            x = ox2
            for y in range(oy1 - room.h + 1, oy2):
                candidates.append((room.w, room.h, x, y, False))
            y = oy1 - room.h
            for x in range(ox1 - room.w + 1, ox2):
                candidates.append((room.w, room.h, x, y, False))
            y = oy2
            for x in range(ox1 - room.w + 1, ox2):
                candidates.append((room.w, room.h, x, y, False))
            rw, rh = room.h, room.w
            x = ox1 - rw
            for y in range(oy1 - rh + 1, oy2):
                candidates.append((rw, rh, x, y, True))
            x = ox2
            for y in range(oy1 - rh + 1, oy2):
                candidates.append((rw, rh, x, y, True))
            y = oy1 - rh
            for x in range(ox1 - rw + 1, ox2):
                candidates.append((rw, rh, x, y, True))
            y = oy2
            for x in range(ox1 - rw + 1, ox2):
                candidates.append((rw, rh, x, y, True))
        uniq = set(); filtered = []
        for w,h,x,y,rot in candidates:
            key = (w,h,int(x),int(y))
            if key in uniq: continue
            uniq.add(key)
            if x < -room.w or y < -room.h or x > self.grid_w or y > self.grid_h: continue
            filtered.append((w,h,int(x),int(y),rot))
            if len(filtered) >= max_candidates: break
        random.shuffle(filtered)
        return filtered

    def _fallback_scan(self, room, limit=5000):
        positions = []
        for y in range(0, self.grid_h - room.h + 1):
            for x in range(0, self.grid_w - room.w + 1):
                positions.append((room.w, room.h, x, y, False))
        random.shuffle(positions)
        return positions[:limit]

    def _compact_pass(self, iterations=3):
        iterations = int(iterations * self.gen_params.compact_strength)
        for it in range(max(1, iterations)):
            order = list(self.placed)
            random.shuffle(order)
            for r in order:
                was_touching = any(r.touches(o) for o in self.placed if o is not r)
                improved = False
                moves = [(-1,0),(0,-1),(1,0),(0,1)]
                random.shuffle(moves)
                for dx,dy in moves:
                    nx, ny = r.x + dx, r.y + dy
                    if not self._can_place(r.w,r.h,nx,ny): continue
                    touching = self._touching_types_if_placed(r.w,r.h,nx,ny)
                    if was_touching and len(touching) == 0: continue
                    for t in touching:
                        if (r.type, t) in FORBIDDEN_ADJ or (t, r.type) in FORBIDDEN_ADJ: break
                    else:
                        cx = nx + r.w/2; cy = ny + r.h/2
                        oldcx = r.x + r.w/2; oldcy = r.y + r.h/2
                        old_dist = math.hypot(oldcx - self.grid_w/2, oldcy - self.grid_h/2)
                        new_dist = math.hypot(cx - self.grid_w/2, cy - self.grid_h/2)
                        if new_dist < old_dist:
                            self.occupancy[r.y:r.y+r.h, r.x:r.x+r.w] = 0
                            r.x, r.y = nx, ny
                            self.occupancy[r.y:r.y+r.h, r.x:r.x+r.w] = 1
                            improved = True
                            break
        return

    def generate(self):
        for room in self.rooms:
            best = None
            best_score = -1e9
            candidates = self._generate_adjacent_candidates(room)
            for w,h,x,y,rot in candidates:
                if not self._can_place(w,h,x,y): continue
                sc = self._score_candidate(room.type, w,h,x,y)
                if sc > best_score:
                    best_score = sc; best = (w,h,x,y,rot,sc)
            if best is None or (len(self.placed)>0 and best_score < -4000):
                fallback = self._fallback_scan(room)
                for w,h,x,y,rot in fallback:
                    if not self._can_place(w,h,x,y): continue
                    sc = self._score_candidate(room.type, w,h,x,y)
                    if sc > best_score:
                        best_score = sc; best = (w,h,x,y,rot,sc)
            if best is not None and best[5] > W_FORBID/2:
                w,h,x,y,rot,sc = best
                room.w, room.h = w,h
                self._place_room(room, x, y)
            else:
                placed_flag = False
                for shrink in (0.9,0.8,0.7):
                    nw = max(MIN_W_CELLS, int(room.w * shrink))
                    nh = max(MIN_H_CELLS, int(room.h * shrink))
                    for y in range(0, self.grid_h - nh + 1):
                        for x in range(0, self.grid_w - nw + 1):
                            if self._can_place(nw,nh,x,y):
                                touching = self._touching_types_if_placed(nw,nh,x,y)
                                bad = any((room.type,t) in FORBIDDEN_ADJ or (t,room.type) in FORBIDDEN_ADJ for t in touching)
                                if bad: continue
                                room.w, room.h = nw, nh
                                self._place_room(room,x,y)
                                placed_flag = True; break
                        if placed_flag: break
                    if placed_flag: break
                if not placed_flag and self.verbose:
                    print(f"[WARN] Could not place room {room.type} ({room.idx})")
        self._compact_pass(iterations=6)
        self._place_doors()
        return self.placed

    def _place_doors(self):
        for a in self.placed:
            for b in self.placed:
                if a is b: continue
                if a.touches(b):
                    if (b.type in PREFERRED_ADJ.get(a.type, [])) or ('Lobby' in (a.type,b.type)) or (a.type=='Dining' and b.type=='Kitchen') or (a.type=='Kitchen' and b.type=='Dining'):
                        ax1,ay1,ax2,ay2 = a.bbox(); bx1,by1,bx2,by2 = b.bbox()
                        if ax2 == bx1:
                            ymid = int(max(ay1,by1) + (min(ay2,by2)-max(ay1,by1))/2)
                            a.doors.append(('V',(ax2,ymid))); b.doors.append(('V',(bx1,ymid)))
                        elif bx2 == ax1:
                            ymid = int(max(ay1,by1) + (min(ay2,by2)-max(ay1,by1))/2)
                            a.doors.append(('V',(ax1,ymid))); b.doors.append(('V',(bx2,ymid)))
                        elif ay2 == by1:
                            xmid = int(max(ax1,bx1) + (min(ax2,bx2)-max(ax1,bx1))/2)
                            a.doors.append(('H',(xmid,ay2))); b.doors.append(('H',(xmid,by1)))
                        elif by2 == ay1:
                            xmid = int(max(ax1,bx1) + (min(ax2,bx2)-max(ax1,bx1))/2)
                            a.doors.append(('H',(xmid,ay1))); b.doors.append(('H',(xmid,by2)))
        for r in self.placed:
            uniq=[]; seen=set()
            for d in r.doors:
                if d not in seen:
                    uniq.append(d); seen.add(d)
            r.doors = uniq

    def render(self, annotated=True, save_path=None):
        canvas = Image.new('RGB', (self.canvas_w, self.canvas_h), (255,255,255))
        draw = ImageDraw.Draw(canvas)
        font = choose_font(14)
        for r in self.placed:
            x_px = cells_to_px(r.x, self.grid_cell); y_px = cells_to_px(r.y, self.grid_cell)
            w_px = cells_to_px(r.w, self.grid_cell); h_px = cells_to_px(r.h, self.grid_cell)
            draw.rectangle([x_px, y_px, x_px + w_px, y_px + h_px], fill=tuple(r.color), outline=tuple(COLOR_MAP['Wall']), width=WALL_THICKNESS_PX)
            if annotated:
                pixels_total = self.canvas_w * self.canvas_h
                pixels_per_sqft = (pixels_total * 0.6) / max(1.0, self.total_sqft)
                area_sqft = (w_px * h_px) / max(1.0, pixels_per_sqft)
                label = f"{r.type}\n{int(area_sqft)} sqft"
                bbox = draw.textbbox((0,0), label, font=font)
                tw = bbox[2]-bbox[0]; th = bbox[3]-bbox[1]
                cx = x_px + w_px//2; cy = y_px + h_px//2
                # Use a transparent white background rectangle for the text
                temp_canvas = Image.new('RGBA', canvas.size)
                temp_draw = ImageDraw.Draw(temp_canvas)
                temp_draw.rectangle([cx-tw//2-4, cy-th//2-4, cx+tw//2+4, cy+th//2+4], fill=(255,255,255,220))
                canvas = Image.alpha_composite(canvas.convert('RGBA'), temp_canvas).convert('RGB')
                draw = ImageDraw.Draw(canvas) # Re-get draw context after composite
                draw.text((cx, cy), label, fill=(20,20,20), font=font, anchor="mm")
            for d in r.doors:
                orient,(cx,cy) = d
                dx = cells_to_px(cx, self.grid_cell); dy = cells_to_px(cy, self.grid_cell)
                if orient == 'V':
                    dh = max(4, self.grid_cell//2)
                    draw.rectangle([dx - self.grid_cell//4, dy - dh//2, dx + self.grid_cell//4, dy + dh//2], fill=(200,180,150))
                else:
                    dw = max(4, self.grid_cell//2)
                    draw.rectangle([dx - dw//2, dy - self.grid_cell//4, dx + dw//2, dy + self.grid_cell//4], fill=(200,180,150))
        
        # Crop to fit placed rooms with margin
        if self.placed:
            min_x = min(r.x for r in self.placed); min_y = min(r.y for r in self.placed)
            max_x = max(r.x + r.w for r in self.placed); max_y = max(r.y + r.h for r in self.placed)
            
            # Convert cells to pixels and add a 40px margin
            px0 = max(0, cells_to_px(min_x, self.grid_cell) - 40)
            py0 = max(0, cells_to_px(min_y, self.grid_cell) - 40)
            px1 = min(self.canvas_w, cells_to_px(max_x, self.grid_cell) + 40)
            py1 = min(self.canvas_h, cells_to_px(max_y, self.grid_cell) + 40)
            
            canvas = canvas.crop((px0, py0, px1, py1))
            
        if save_path:
            canvas.save(save_path)
            
        return canvas

    def summary(self):
        pixels_total = self.canvas_w * self.canvas_h
        pixels_per_sqft = (pixels_total * 0.6) / max(1.0, self.total_sqft)
        totalsq = 0.0
        counts = defaultdict(lambda: {'count':0, 'sqft':0.0})
        for r in self.placed:
            w_px = cells_to_px(r.w, self.grid_cell); h_px = cells_to_px(r.h, self.grid_cell)
            sqft = (w_px * h_px) / max(1.0, pixels_per_sqft)
            counts[r.type]['count'] += 1; counts[r.type]['sqft'] += sqft; totalsq += sqft
        return counts, totalsq

    def export_json(self):
        out = {'rooms': [], 'grid': {'w':self.grid_w, 'h':self.grid_h, 'cell_px':self.grid_cell, 'canvas_w':self.canvas_w, 'canvas_h':self.canvas_h}, 'gen_params': str(self.gen_params)}
        for r in self.placed:
            out['rooms'].append({'type': r.type, 'id': r.idx, 'x_cells': r.x, 'y_cells': r.y, 'w_cells': r.w, 'h_cells': r.h})
        return out

# -------------------------
# Streamlit UI & Helper Functions (MODIFIED for Quality)
# -------------------------
st.set_page_config(page_title="Generative Adjacency Tetris", layout="wide")

# Helper to convert PIL image to bytes with explicit quality/format
def pil_to_bytes(pil_img, fmt='PNG'):
    bio = BytesIO()
    if fmt == 'PNG':
        # PNG is lossless, so no quality parameter, but ensures high quality
        pil_img.save(bio, format=fmt)
    elif fmt == 'JPEG':
        # Use high quality for JPEG (max 95 is standard for PIL high quality)
        pil_img.save(bio, format=fmt, quality=95)
    else:
        pil_img.save(bio, format=fmt)
        
    bio.seek(0)
    return bio.read()

def generate_svg_from_json(plan_json):
    grid = plan_json['grid']
    # Use a fixed, large canvas size for SVG export for better vector scaling
    cw = grid.get('canvas_w', 1200) # Ensure a reasonable width for SVG viewbox
    ch = grid.get('canvas_h', 800)
    cell = grid['cell_px']
    svg_parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 {cw} {ch}">']
    svg_parts.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    for r in plan_json['rooms']:
        x = r['x_cells'] * cell; y = r['y_cells'] * cell; w = r['w_cells'] * cell; h = r['h_cells'] * cell
        color = COLOR_MAP.get(r['type'], [200,200,200])
        fill = f'rgb({color[0]},{color[1]},{color[2]})'
        svg_parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="#000" stroke-width="3" rx="8" ry="8"/>')
        # Add a text background for better contrast in vector export
        svg_parts.append(f'<rect x="{x + w/2 - 100}" y="{y + h/2 - 20}" width="200" height="40" fill="rgba(255,255,255,0.8)" rx="5" ry="5"/>')
        svg_parts.append(f'<text x="{x + w/2}" y="{y + h/2}" font-family="Arial, sans-serif" font-size="24" font-weight="bold" text-anchor="middle" dominant-baseline="central" fill="#111">{r["type"]}</text>')
    svg_parts.append('</svg>')
    return '\n'.join(svg_parts).encode('utf-8')


# Google-like Material/Clean CSS + white section headings + overlay styles
st.markdown("""
<style>
/* Base Streamlit overrides */
.stApp {background-color: #f2f4f8;} /* subtle off-white */
.main > div {padding-top: 1.6rem;}

/* Material Card Style */
.section {
    padding: 18px;
    border-radius: 12px;
    background: #ffffff;
    box-shadow: 0 6px 18px rgba(20, 30, 40, 0.06);
    margin-bottom: 18px;
}

/* Header/Title Styling */
.header {display:flex; align-items:center; gap:12px; margin-bottom: 18px;}
.h1 {font-size: 28px; font-weight: 700; color: #0f1724; padding-bottom: 4px;}
.sub {color: #6b7280; font-size: 14px; font-weight: 500;}

/* Button & Primary Color - Google Blue */
.stButton>button {
    background-color: #1a73e8; /* Google Blue */
    color: white;
    font-weight: 600;
    border-radius: 8px;
    border: 1px solid #1a73e8;
    transition: background-color 0.18s ease;
}
.stButton>button:hover {
    background-color: #165ec9;
    border-color: #165ec9;
}

/* Chat Box / Info Box Styling */
.chat-box {
    min-height: 200px;
    max-height: 420px;
    overflow-y: auto;
    border: 1px solid #e6eef6;
    padding: 16px;
    border-radius: 8px;
    background-color: #fbfdff;
}
.stTextInput>div>div>input, .stNumberInput>div>div>input {
    border-radius: 8px;
}

/* Thumbnails */
.plan-thumb-container {
    padding: 6px;
    border: 1px solid #e8f0ff;
    border-radius: 8px;
    transition: box-shadow 0.2s, border 0.2s, transform 0.08s;
    background: linear-gradient(180deg, #ffffff, #fbfcff);
}
.plan-thumb-container:hover {
    box-shadow: 0 10px 30px rgba(12, 46, 99, 0.06);
    transform: translateY(-2px);
}
.selected-thumb {
    border: 3px solid #1a73e8; /* Highlight for selected plan */
}

/* White section title helper for inline HTML headings */
.white-title { color: white !important; background: linear-gradient(90deg, #1a73e8, #2f80ed); padding:8px 12px; border-radius:8px; display:inline-block; font-weight:600; }

/* Overlay (lightbox/modal) for enlarged image */
.overlay {
    position: fixed;
    left: 0; top: 0; right: 0; bottom: 0;
    z-index: 9999;
    display:flex;
    align-items:center;
    justify-content:center;
    background: rgba(10, 20, 30, 0.6);
}
.overlay-card {
    width: 88%;
    max-width: 1400px;
    border-radius: 12px;
    background: #ffffff;
    padding: 14px;
    box-shadow: 0 20px 50px rgba(2,7,23,0.55);
}
.overlay-controls { display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:8px; }
.small-btn {
    background:#1a73e8; color:white; padding:8px 12px; border-radius:8px; border:none; cursor:pointer;
}
@media (max-width: 900px) {
    .overlay-card { width: 96%; padding:10px; }
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# UI Layout
# -------------------------

# Top header
st.markdown('<div class="header">', unsafe_allow_html=True)
st.markdown('<div class="h1">📐 Generative Floor Planner</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">AI-Assisted Layout Design via Adjacency-Tetris</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

left, right = st.columns([3,7])

# Session state initialization
if 'plans' not in st.session_state:
    st.session_state.plans = []
if 'selected' not in st.session_state:
    st.session_state.selected = 0
if 'enlarged_idx' not in st.session_state:
    st.session_state.enlarged_idx = None
if 'seed_in' not in st.session_state:
    st.session_state.seed_in = ""
if 'bhk_sel' not in st.session_state:
    st.session_state.bhk_sel = 3
if 'area_in' not in st.session_state:
    st.session_state.area_in = 1500

# -------------------------
# Left Column (Configuration + Generate)
# -------------------------
with left:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    # Configuration title using white-title class
    st.markdown('<div style="display:flex; justify-content:space-between; align-items:center;"><div class="white-title">Configuration</div></div>', unsafe_allow_html=True)
    st.write("")  # spacing

    col_bhk, col_area = st.columns(2)
    with col_bhk:
        bhk = st.selectbox('', options=[1,2,3,4], index=st.session_state.bhk_sel-1, format_func=lambda x: f"{x} BHK", key='bhk_sel')
    with col_area:
        total_area = st.number_input('', value=st.session_state.area_in, min_value=200, step=50, key='area_in')

    # Optional seed label converted to white text + input without repeated label
    st.markdown('<div style="margin-top:10px;"><span class="white-title" style="font-size:12px; padding:6px 8px;">Optional Seed (Randomize if empty)</span></div>', unsafe_allow_html=True)
    seed = st.text_input('', value=st.session_state.seed_in, placeholder='Enter seed (leave blank for random)', key='seed_in')

    st.write('') # Add vertical space
    generate_btn = st.button('Generate 5 Plans ✨', use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Log & Tips with white heading
    st.markdown('<div class="section" style="margin-top:12px">', unsafe_allow_html=True)
    st.markdown('<div class="white-title" style="font-size:14px; padding:8px 10px;">Log & Tips</div>', unsafe_allow_html=True)
    st.markdown('<div style="margin-top:10px">', unsafe_allow_html=True)
    st.markdown('<div class="chat-box" id="chat-box">', unsafe_allow_html=True)
    st.info('**Welcome!** Set your desired BHK and area, then hit **"Generate 5 Plans"** to start the floor-planning process. Plans are generated with adjacency and compactness prioritized.', icon='ℹ️')
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Generation Logic (runs immediately when generate_btn is clicked)
# placed right after the configuration UI so it feels cohesive
# -------------------------
if generate_btn:
    # reset enlarge state & plans
    st.session_state.enlarged_idx = None
    st.session_state.plans = []
    st.session_state.selected = 0
    with st.spinner('Generating floor plans...'):
        for i in range(5):
            s = None
            if st.session_state.seed_in and st.session_state.seed_in.strip():
                try:
                    s = int(st.session_state.seed_in) + i
                except:
                    s = abs(hash(st.session_state.seed_in)) % 1000000 + i

            gen_params = GenerativeParams(seed=s)
            engine = GenerativeAdjacencyTetris(
                bhk=st.session_state.bhk_sel,
                total_sqft=st.session_state.area_in,
                canvas_w=CANVAS_W_PX,
                canvas_h=CANVAS_H_PX,
                grid_cell=GRID_CELL_PX,
                gen_params=gen_params,
                verbose=False
            )
            engine.generate()

            # The PIL image object (pil_img) now holds the full-size, cropped, high-quality rendering
            pil_img = engine.render(annotated=True)
            j = engine.export_json()

            # Generate bytes for download/display
            svg_b = generate_svg_from_json(j)
            png_b = pil_to_bytes(pil_img, fmt='PNG')

            # Store the full PIL object, not a resized version
            st.session_state.plans.append({
                'engine': engine,
                'pil': pil_img,
                'json': j,
                'svg': svg_b,
                'png': png_b,
                'seed': gen_params.seed
            })

        st.success('Generation complete — 5 plans ready for inspection!')

# -------------------------
# Right column: Viewer + Thumbnails + Inspector
# -------------------------
with right:
    # Display overlay (enlarged image) if requested
    if st.session_state.enlarged_idx is not None:
        idx = st.session_state.enlarged_idx
        if 0 <= idx < len(st.session_state.plans):
            plan = st.session_state.plans[idx]
            # create overlay using raw HTML and base64 image (so it floats above)
            png_b = plan['png']
            b64 = base64.b64encode(png_b).decode('utf-8')
            # Using HTML overlay (clicking close sets session state via a Streamlit button below)
            st.markdown(
                f'''
                <div class="overlay">
                  <div class="overlay-card">
                    <div class="overlay-controls">
                      <div style="font-weight:700;">Enlarged — Plan {idx+1} (Seed: {plan["seed"]})</div>
                      <div>
                        <form action="javascript:void(0);">
                          <button id="close-btn" class="small-btn" onclick="document.querySelector('#close-link').click();">Close</button>
                        </form>
                      </div>
                    </div>
                    <div style="text-align:center;">
                      <img src="data:image/png;base64,{b64}" style="max-width:100%; height:auto; border-radius:8px;"/>
                    </div>
                  </div>
                </div>
                ''',
                unsafe_allow_html=True
            )
            # hidden link to a Streamlit button (a small trick to let user close via a Streamlit button)
            if st.button("Close Enlarged View", key=f"close_enlarge_{idx}"):
                st.session_state.enlarged_idx = None

    # Viewer title with white badge
    st.markdown('<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;"><div class="white-title" style="font-size:14px;">Generated Plans (Select to Inspect)</div></div>', unsafe_allow_html=True)

    if st.session_state.plans:
        plans = st.session_state.plans

        # Thumbnails row
        st.markdown('<div class="section">', unsafe_allow_html=True)
        cols = st.columns(5)
        for i, plan in enumerate(plans):
            is_selected = st.session_state.selected == i
            style_class = "plan-thumb-container selected-thumb" if is_selected else "plan-thumb-container"
            with cols[i]:
                st.markdown(f'<div class="{style_class}">', unsafe_allow_html=True)
                # Show thumbnail using the actual PIL image (streamlit will resize in the column)
                st.image(plan['pil'], use_column_width=True, caption=f'Plan {i+1}', clamp=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Two actions: select/inspect (left) and enlarge (right)
                col_a, col_b = st.columns([2,1])
                with col_a:
                    if st.button(f'Inspect {i+1}', key=f'inspect_{i}'):
                        st.session_state.selected = i
                        st.session_state.enlarged_idx = None  # close any overlay
                with col_b:
                    # Enlarge button opens overlay
                    if st.button('View', key=f'view_{i}'):
                        st.session_state.enlarged_idx = i
                        st.session_state.selected = i
        st.markdown('</div>', unsafe_allow_html=True)

        # Inspector area below thumbnails
        sel = st.session_state.get('selected', 0)
        selected = plans[sel]

        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown(f'<div style="display:flex; justify-content:space-between; align-items:center;"><div style="font-weight:700; font-size:16px;">Plan {sel+1} Details</div><div style="color:#6b7280;">Seed: {selected["seed"]}</div></div>', unsafe_allow_html=True)
        cols = st.columns([5,3])

        with cols[0]:
            st.image(selected['pil'], use_column_width=True)

        with cols[1]:
            st.markdown("#### Summary & Export")
            counts, totalsq = selected['engine'].summary()

            # Display summary
            for t,d in counts.items():
                st.markdown(f"**{t}:** **{d['count']}** units | ≈ **{int(d['sqft'])}** sqft")

            st.markdown('---')
            st.markdown(f"**Total Area Used:** ≈ **{int(totalsq)}** sqft")

            st.write('') # Spacer
            st.caption(f"Grid: {selected['json']['grid']['w']} x {selected['json']['grid']['h']} | Cell: {selected['json']['grid']['cell_px']} px")

            # Download buttons side-by-side
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.download_button('Download SVG (Vector)', data=selected['svg'], file_name=f'plan_{sel+1}.svg', mime='image/svg+xml', use_container_width=True)
            with d_col2:
                st.download_button('Download PNG', data=selected['png'], file_name=f'plan_{sel+1}.png', mime='image/png', use_container_width=True)
            st.download_button('Export JSON Data', data=json.dumps(selected['json'], indent=2), file_name=f'plan_{sel+1}.json', mime='application/json', use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info('No plans generated yet. Use the Configuration panel on the left and click "Generate 5 Plans".')

# Footer
st.markdown("""
<div style="text-align:center; padding:18px; color:#6b7280; font-size:12px; margin-top: 18px;">
    Powered by Generative Adjacency-Tetris Engine • Modern White/Material Theme
</div>
""", unsafe_allow_html=True)
