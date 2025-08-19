import argparse, os, sys, csv, io, re
import zstandard as zstd
import chess.pgn
from tqdm import tqdm

def open_pgn_stream(path):
    if path == "-":
        return sys.stdin
    if path.endswith(".zst"):
        dctx = zstd.ZstdDecompressor()
        fh = open(path, "rb")
        return io.TextIOWrapper(dctx.stream_reader(fh), encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")

def elo_ok(tags, min_elo: int, allow_missing: bool):
    def parse(v):
        try:
            return int(v)
        except Exception:
            return None
    w = parse(tags.get("WhiteElo", ""))
    b = parse(tags.get("BlackElo", ""))
    if w is None or b is None:
        return allow_missing
    return w >= min_elo and b >= min_elo

def tc_is_bullet(tags):
    tc = tags.get("TimeControl", "")
    # lichess/PGN format like "60+0" (base+inc) in seconds; bullet usually base < 60 (heuristic)
    m = re.match(r"^(\d+)(?:\+\d+)?$", tc.strip())
    if m:
        base = int(m.group(1))
        return base < 60
    # if unknown, don't mark as bullet
    return False

def iter_games_from_inputs(inputs):
    for path in inputs:
        with open_pgn_stream(path) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                yield game

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", help="List of .pgn or .pgn.zst files (use '-' for stdin).", default=[])
    ap.add_argument("--stdin", action="store_true", help="Read PGN stream from stdin (zstdcat file.pgn.zst | ...)")
    ap.add_argument("--out", required=True, help="Output CSV path with columns: fen,uci")
    ap.add_argument("--min-elo", type=int, default=0, help="Keep games where both players Elo >= this value.")
    ap.add_argument("--allow-missing-elo", action="store_true", help="Include games missing Elo tags.")
    ap.add_argument("--exclude-bullet", action="store_true", help="Filter out bullet by heuristic (TimeControl base < 60s).")
    ap.add_argument("--max-games", type=int, default=0, help="Optional limit on number of games processed.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_f = open(args.out, "w", newline="")
    writer = csv.writer(out_f)
    writer.writerow(["fen","uci"])

    def game_iter():
        if args.stdin:
            with open_pgn_stream("-") as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    yield game
        else:
            yield from iter_games_from_inputs(args.inputs)

    n_games = 0
    n_rows = 0
    for game in tqdm(game_iter(), desc="games"):
        tags = game.headers
        if (args.min_elo > 0 or not args.allow_missing_elo) and not elo_ok(tags, args.min_elo, args.allow_missing_elo):
            continue
        if args.exclude_bullet and tc_is_bullet(tags):
            continue

        board = game.board()
        for move in game.mainline_moves():
            fen_before = board.fen()
            uci = move.uci()
            writer.writerow([fen_before, uci])
            n_rows += 1
            board.push(move)

        n_games += 1
        if args.max_games and n_games >= args.max_games:
            break

    out_f.close()
    print(f"Wrote {n_rows} rows from {n_games} games → {args.out}")

if __name__ == "__main__":
    main()
