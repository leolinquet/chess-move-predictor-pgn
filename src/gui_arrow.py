import argparse, math, heapq, time
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, Response, make_response
import chess
import chess.svg as chess_svg

from .model import SmallConvPolicy
from .encoder import board_to_planes
from .move_vocab import compose_uci, PROMO_TO_IDX

# -------- Inference --------
@dataclass
class Predictor:
    model: SmallConvPolicy
    device: torch.device

    @torch.no_grad()
    def legal_distribution(self, board: chess.Board):
        """Probs over legal moves. p(move)=p_from*f * p_to*t * p_promo*p normalized to 1."""
        x = torch.from_numpy(board_to_planes(board)).unsqueeze(0).to(self.device)
        out = self.model(x)
        pf = F.softmax(out["from"],  dim=1).squeeze(0)   # [64]
        pt = F.softmax(out["to"],    dim=1).squeeze(0)   # [64]
        pp = F.softmax(out["promo"], dim=1).squeeze(0)   # [5]

        items = []
        total = 0.0
        for m in board.legal_moves:
            f = int(m.from_square)
            t = int(m.to_square)
            promo_idx = 0
            if m.promotion:
                promo_idx = PROMO_TO_IDX[{chess.QUEEN:'q', chess.ROOK:'r', chess.BISHOP:'b', chess.KNIGHT:'n'}[m.promotion]]
            p = float(pf[f]) * float(pt[t]) * float(pp[promo_idx])
            items.append([m.uci(), p])
            total += p

        if total <= 0:  # no legal moves
            return []
        items = [(uci, p/total) for uci, p in items]
        items.sort(key=lambda x: x[1], reverse=True)
        return items

# -------- App --------
def make_app(ckpt_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallConvPolicy().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    predictor = Predictor(model, device)
    board = chess.Board()

    app = Flask(__name__)

    # --- HTML (no external JS/CSS) ---
    PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Chess Predictor (Local GUI)</title>
  <style>
    :root { --bg:#0a0f1d; --card:#121a2b; --border:#22304f; --text:#e7ecf3; --muted:#9fb1d1; --brand:#00A3FF; }
    html,body { height:100%; margin:0; background:var(--bg); color:var(--text); font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
    .wrap { min-height:100%; display:grid; place-items:center; padding:24px; }
    .card { background:linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.00));
            border:1px solid var(--border); border-radius:16px; box-shadow:0 12px 40px rgba(0,0,0,.35);
            padding:18px; display:flex; gap:24px; align-items:flex-start; }
    .board-wrap { position:relative; width:560px; max-width:70vw; }
    #boardImg { width:100%; height:auto; display:block; border-radius:12px; }
    .overlay { position:absolute; inset:0; display:grid; grid-template-columns:repeat(8,1fr); grid-template-rows:repeat(8,1fr); }
    .sq { cursor:pointer; }
    .sq.sel { outline:3px solid var(--brand); outline-offset:-3px; }
    .controls { display:flex; flex-direction:column; gap:12px; min-width:300px; }
    .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    button { background:#172243; color:var(--text); border:1px solid var(--border); padding:10px 12px; border-radius:10px; cursor:pointer; }
    button:hover { filter:brightness(1.1); }
    .pill { background:#0f1831; border:1px solid var(--border); padding:8px 10px; border-radius:999px; color:var(--muted); }
    ul { margin:0; padding-left:18px; max-height:240px; overflow:auto; }
    .pct { color:var(--brand); }
    .title { font-size:18px; font-weight:600; }
    .muted { color:var(--muted); }
    .sep { height:1px; background:var(--border); margin:8px 0; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="board-wrap">
        <img id="boardImg" src="/board.svg?ts=0" alt="board"/>
        <div id="overlay" class="overlay"></div>
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
        <div class="muted">Click two squares to move. Promotions will ask what to promote to.</div>
      </div>
    </div>
  </div>

  <script>
    const files = 'abcdefgh';
    function idxToSq(i){ const f=i%8, r=Math.floor(i/8); return files[f] + (r+1); }
    function sqAt(col,row){ return files[col] + (row+1); } // col 0..7 left->right, row 1..8 bottom->top
    function refreshBoard(){ document.getElementById('boardImg').src='/board.svg?ts=' + Date.now(); }

    async function api(path, method='GET', body=null){
      const res = await fetch(path, { method, headers:{'Content-Type':'application/json'}, body: body?JSON.stringify(body):null });
      return res.json();
    }

    function updateTop(sugg, cands){
      const pill = document.getElementById('suggestPill');
      if(!sugg){ pill.textContent = 'No suggestion'; } else {
        const pct = (sugg[1]*100).toFixed(1);
        pill.textContent = 'Suggested: ' + sugg[0] + ' (' + pct + '%)';
      }
      const ul = document.getElementById('candList'); ul.innerHTML='';
      (cands||[]).slice(0,5).forEach(([u,p])=>{
        const li=document.createElement('li'); li.innerHTML = u + ' <span class="pct">(' + (p*100).toFixed(1) + '%)</span>';
        ul.appendChild(li);
      });
    }

    let LEGAL = [];  // list of UCIs from /api/state
    async function refresh(){
      const s = await api('/api/state');
      updateTop(s.suggested, s.candidates);
      LEGAL = s.legal || [];
      refreshBoard();
    }

    function buildOverlay(){
      const ov = document.getElementById('overlay'); ov.innerHTML='';
      // 8x8 clickable cells; map row visually so a1 bottom-left.
      for(let r=8; r>=1; --r){
        for(let c=0; c<8; ++c){
          const div=document.createElement('div');
          const sq = files[c] + r;
          div.className='sq';
          div.dataset.sq=sq;
          ov.appendChild(div);
        }
      }
    }

    function choosePromo(from,to){
      // See if there are promotion legals; if so, ask which one.
      const options = LEGAL.filter(u=>u.startsWith(from+to) && u.length===5).map(u=>u[4]);
      if(options.length===0) return '';  // not a promotion
      const allowed = ['q','r','b','n'];
      let pick = prompt('Promote to (q,r,b,n)?','q');
      if(!allowed.includes(pick)) pick = 'q';
      return pick;
    }

    function clickToMove(){
      const ov = document.getElementById('overlay');
      let from = null;
      ov.addEventListener('click', async (e)=>{
        const cell = e.target.closest('.sq'); if(!cell) return;
        const sq = cell.dataset.sq;

        if(!from){
          from = sq; cell.classList.add('sel');
          return;
        }else{
          // find selected div to unselect
          const sel = ov.querySelector('.sq.sel'); if(sel) sel.classList.remove('sel');
          const promo = choosePromo(from, sq);
          const uci = from + sq + promo;
          from = null;

          const res = await api('/api/move','POST',{uci});
          if(!res.ok){ alert('Illegal move'); }
          await refresh();
        }
      });
    }

    async function main(){
      buildOverlay();
      clickToMove();
      document.getElementById('btnPlay').onclick = async ()=>{ await api('/api/play_suggested','POST',{}); await refresh(); };
      document.getElementById('btnUndo').onclick = async ()=>{ await api('/api/undo','POST',{}); await refresh(); };
      document.getElementById('btnNew').onclick = async ()=>{ await api('/api/new','POST',{}); await refresh(); };
      await refresh();
      // keep board responsive on resize (image gets new size; overlay tracks since it's absolute)
      window.addEventListener('resize', refreshBoard);
    }
    window.addEventListener('load', main);
  </script>
</body>
</html>
"""

    @app.get("/")
    def home():
        return Response(PAGE, mimetype="text/html")

    @app.get("/board.svg")
    def board_svg():
        # compute suggestion + arrow
        dist = predictor.legal_distribution(board)
        arrow = None
        if dist:
            top_uci = dist[0][0]
            mv = chess.Move.from_uci(top_uci)
            arrow = chess_svg.Arrow(mv.from_square, mv.to_square, color="#00A3FF")

        svg = chess_svg.board(
            board,
            arrows=[arrow] if arrow else [],
            coordinates=True,
            lastmove=board.peek() if board.move_stack else None,
            size=560,
        )
        resp = make_response(svg)
        resp.headers["Content-Type"] = "image/svg+xml"
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        return resp

    @app.get("/api/state")
    def api_state():
        dist = predictor.legal_distribution(board)
        data = dict(
            fen=board.fen(),
            suggested=dist[0] if dist else None,
            candidates=dist[:5] if dist else [],
            legal=[m.uci() for m in board.legal_moves],
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    app = make_app(args.ckpt)
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()
