import argparse, math, heapq, json
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, Response
import chess

from .model import SmallConvPolicy
from .encoder import board_to_planes
from .move_vocab import compose_uci, PROMO_TO_IDX

# ---------- Inference ----------
@dataclass
class Predictor:
    model: SmallConvPolicy
    device: torch.device

    @torch.no_grad()
    def legal_distribution(self, board: chess.Board):
        """Return probs over ALL legal moves. p(move)=p_from*f * p_to*t * p_promo*p normalized to 1."""
        x = torch.from_numpy(board_to_planes(board)).unsqueeze(0).to(self.device)
        out = self.model(x)
        pf = F.softmax(out["from"],  dim=1).squeeze(0)   # [64]
        pt = F.softmax(out["to"],    dim=1).squeeze(0)   # [64]
        pp = F.softmax(out["promo"], dim=1).squeeze(0)   # [5]

        items = []
        total = 0.0
        for m in board.legal_moves:
            f = int(m.from_square)         # 0..63 (python-chess a1=0)
            t = int(m.to_square)           # 0..63
            promo_idx = 0
            if m.promotion:
                promo_idx = PROMO_TO_IDX[{chess.QUEEN:'q', chess.ROOK:'r', chess.BISHOP:'b', chess.KNIGHT:'n'}[m.promotion]]
            p = float(pf[f]) * float(pt[t]) * float(pp[promo_idx])
            uci = m.uci()
            items.append([uci, p])
            total += p

        # normalize to 1
        if total <= 0:
            return []
        items = [(uci, p/total) for uci, p in items]
        items.sort(key=lambda x: x[1], reverse=True)
        return items

# ---------- Web app ----------
def make_app(ckpt: str, legal_only: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallConvPolicy().to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    predictor = Predictor(model, device)
    board = chess.Board()

    app = Flask(__name__)

    @app.get("/")
    def home():
        # Inline HTML/CSS/JS (uses chessboard.js CDN)
        html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Chess Predictor</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
  <style>
    :root {{
      --bg:#0a0f1d; --card:#121a2b; --border:#22304f; --text:#e7ecf3; --muted:#9fb1d1; --brand:#00A3FF;
    }}
    html,body {{ height:100%; margin:0; overflow:hidden; background:var(--bg); color:var(--text); font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
    .wrap {{ height:100%; display:grid; place-items:center; }}
    .card {{
      background:linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.00));
      border:1px solid var(--border); border-radius:16px; box-shadow:0 12px 40px rgba(0,0,0,.35);
      padding:18px; display:flex; gap:24px; align-items:flex-start;
    }}
    .left {{ position:relative; }}
    #board {{ width:560px; max-width:70vw; aspect-ratio:1/1; }}
    #arrowLayer {{
      position:absolute; inset:0; pointer-events:none;
    }}
    .controls {{ display:flex; flex-direction:column; gap:12px; min-width:280px; }}
    .row {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; }}
    button {{ background:#172243; color:var(--text); border:1px solid var(--border); padding:10px 12px; border-radius:10px; cursor:pointer; }}
    button:hover {{ filter:brightness(1.1); }}
    .pill {{ background:#0f1831; border:1px solid var(--border); padding:8px 10px; border-radius:999px; color:var(--muted); }}
    ul {{ margin:0; padding-left:18px; max-height:180px; overflow:auto; }}
    .pct {{ color:var(--brand); }}
    .title {{ font-size:18px; font-weight:600; }}
    .muted {{ color:var(--muted); }}
    .sep {{ height:1px; background:var(--border); margin:8px 0; }}
    .footer {{ font-size:12px; color:var(--muted); }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="left">
        <div id="board"></div>
        <svg id="arrowLayer"></svg>
      </div>
      <div class="controls">
        <div class="title">Model suggestion</div>
        <div id="suggestPill" class="pill">Loading…</div>
        <div class="row">
          <button id="btnPlay">Play suggested</button>
          <button id="btnUndo">Undo</button>
          <button id="btnNew">New game</button>
        </div>
        <div class="sep"></div>
        <div class="title">Top-5 candidates</div>
        <ul id="candList"></ul>
        <div class="sep"></div>
        <div class="footer muted">Drag pieces to move. Promotions will ask what to promote to.</div>
      </div>
    </div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/chess.js@0.13.4/chess.min.js" defer>window.addEventListener('load', main);
  </script>
  <script src="https://cdn.jsdelivr.net/npm/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js" defer></script>
  <script>
    const boardDiv = document.getElementById('board');
    let game = new Chess(); // client copy (server is source of truth)
    let board = null;
    let currentSuggestion = null;

    async function api(path, method='GET', body=null) {{
      const res = await fetch(path, {{
        method, headers: {{'Content-Type':'application/json'}},
        body: body ? JSON.stringify(body) : null
      }});
      return res.json();
    }}

    function sqToCoords(sq, size) {{
      // 'a1' bottom-left in display; chessboard.js draws a8 at top-left
      const files='abcdefgh';
      const f = files.indexOf(sq[0]);
      const r = parseInt(sq[1],10); // 1..8
      const cell = size/8;
      const x = f*cell + cell/2;
      const y = (8-r)*cell + cell/2;
      return [x,y];
    }}

    function drawArrow(from, to) {{
      const svg = document.getElementById('arrowLayer');
      const size = boardDiv.clientWidth;
      svg.setAttribute('viewBox', '0 0 ' + size + ' ' + size);
      svg.setAttribute('width', size);
      svg.setAttribute('height', size);
      svg.innerHTML = '';
      if (!from || !to) return;

      const [x1,y1] = sqToCoords(from, size);
      const [x2,y2] = sqToCoords(to, size);

      const line = document.createElementNS('http://www.w3.org/2000/svg','line');
      line.setAttribute('x1', x1); line.setAttribute('y1', y1);
      line.setAttribute('x2', x2); line.setAttribute('y2', y2);
      line.setAttribute('stroke', '#00A3FF'); line.setAttribute('stroke-width', 10);
      line.setAttribute('stroke-linecap', 'round'); line.setAttribute('opacity','0.85');

      const arrow = document.createElementNS('http://www.w3.org/2000/svg','polygon');
      const ang = Math.atan2(y2-y1, x2-x1);
      const head = 20, width = 14;
      const hx = Math.cos(ang), hy = Math.sin(ang);
      const p1x = x2, p1y = y2;
      const p2x = x2 - head*hx + width*hy, p2y = y2 - head*hy - width*hx;
      const p3x = x2 - head*hx - width*hy, p3y = y2 - head*hy + width*hx;
      arrow.setAttribute('points', p1x+','+p1y+' '+p2x+','+p2y+' '+p3x+','+p3y);
      arrow.setAttribute('fill', '#00A3FF'); arrow.setAttribute('opacity','0.95');

      svg.appendChild(line); svg.appendChild(arrow);
    }}

    function updateCandidates(cands) {{
      const ul = document.getElementById('candList');
      ul.innerHTML = '';
      cands.forEach(([uci, p]) => {{
        const li = document.createElement('li');
        const pct = (p*100).toFixed(1);
        li.innerHTML = `${{uci}} <span class="pct">(${{pct}}%)</span>`;
        ul.appendChild(li);
      }});
    }}

    function setSuggestionText(sugg) {{
      const pill = document.getElementById('suggestPill');
      if (!sugg) {{
        pill.textContent = 'No suggestion';
        return;
      }}
      const pct = (sugg[1]*100).toFixed(1);
      pill.textContent = `Suggested: ${{sugg[0]}}  (${{pct}}%)`;
    }}

    async function refresh() {{
      const data = await api('/api/state');
      game.load(data.fen);
      board.position(data.fen, false);
      currentSuggestion = data.suggested || null;
      setSuggestionText(currentSuggestion);
      updateCandidates(data.candidates || []);
      if (currentSuggestion) {{
        drawArrow(currentSuggestion[0].slice(0,2), currentSuggestion[0].slice(2,4));
      }} else {{
        drawArrow(null,null);
      }}
    }}

    function choosePromotion(from, to) {{
      // simple prompt; could be upgraded to a modal
      const legal = game.moves({{ verbose: true }}).filter(m => m.from===from && m.to===to && m.promotion);
      if (!legal.length) return '';
      const choice = prompt("Promote to (q,r,b,n)?", "q");
      const allowed = ['q','r','b','n'];
      return allowed.includes(choice) ? choice : 'q';
    }}

    async function onDrop (source, target, piece, newPos, oldPos, orientation) {{
      // Build a UCI and ask server to play it
      let promo = choosePromotion(source, target);
      const uci = source + target + promo;
      const res = await api('/api/move', 'POST', {{uci}});
      if (!res.ok) {{
        // illegal on server; snapback
        return 'snapback';
      }}
      await refresh();
    }}

    window.addEventListener('resize', () => {{
      board.resize();
      if (currentSuggestion) {{
        drawArrow(currentSuggestion[0].slice(0,2), currentSuggestion[0].slice(2,4));
      }}
    }});

    if (typeof Chessboard==='undefined'){document.getElementById('suggestPill').textContent='Failed to load chessboard.js (CDN blocked)';}
    async function main() {{
      board = Chessboard('board', {{
        position: 'start',
        draggable: true,
        dropOffBoard: 'snapback',
        onDrop: onDrop,
      }});
      await refresh();

      document.getElementById('btnPlay').onclick = async () => {{
        await api('/api/play_suggested','POST',{{}});
        await refresh();
      }};
      document.getElementById('btnUndo').onclick = async () => {{
        await api('/api/undo','POST',{{}});
        await refresh();
      }};
      document.getElementById('btnNew').onclick = async () => {{
        await api('/api/new','POST',{{}});
        await refresh();
      }};
    }}
    main();
  </script>
</body>
</html>
"""
        return Response(html, mimetype="text/html")

    @app.get("/api/state")
    def api_state():
        dist = predictor.legal_distribution(board)
        suggested = dist[0] if dist else None
        # Only top 5 to keep UI light
        data = dict(
            fen=board.fen(),
            suggested=suggested,
            candidates=dist[:5] if dist else [],
        )
        return jsonify(data)

    @app.post("/api/move")
    def api_move():
        data = request.get_json(force=True)
        uci = data.get("uci","")
        try:
            m = chess.Move.from_uci(uci)
        except Exception:
            return jsonify(ok=False, msg="bad-uci")
        if m not in board.legal_moves:
            return jsonify(ok=False, msg="illegal")
        board.push(m)
        return jsonify(ok=True)

    @app.post("/api/undo")
    def api_undo():
        if board.move_stack:
            board.pop()
        return jsonify(ok=True)

    @app.post("/api/new")
    def api_new():
        board.reset()
        return jsonify(ok=True)

    @app.post("/api/play_suggested")
    def api_play_suggested():
        dist = predictor.legal_distribution(board)
        if dist:
            m = chess.Move.from_uci(dist[0][0])
            if m in board.legal_moves:
                board.push(m)
        return jsonify(ok=True)

    return app

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to runs/.../best.pt")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    app = make_app(args.ckpt, legal_only=True)
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()

