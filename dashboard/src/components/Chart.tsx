import { useEffect, useRef, useState } from "react";

export type Bar = { t: string; o: number; h: number; l: number; c: number; v: number };

// Canvas candlestick chart: wheel to zoom, drag to pan, crosshair with OHLCV
// readout. DPR-aware; draws only the visible window so 15y of daily bars pans
// at 60fps.
export default function Chart({ bars, live }: { bars: Bar[]; live?: number | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [view, setView] = useState({ start: 0, count: 0 });
  const [cross, setCross] = useState<{ x: number; y: number } | null>(null);
  const drag = useRef<{ x: number; start: number } | null>(null);

  useEffect(() => {
    const count = Math.min(bars.length, 180);
    setView({ start: Math.max(0, bars.length - count), count });
  }, [bars]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !bars.length || !view.count) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = 420;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.height = `${h}px`;
    const ctx = canvas.getContext("2d")!;
    ctx.scale(dpr, dpr);

    const padR = 64, padB = 22, volH = 56;
    const plotW = w - padR, plotH = h - padB - volH;
    const vis = bars.slice(view.start, view.start + view.count);
    if (!vis.length) return;

    let lo = Infinity, hi = -Infinity, vMax = 0;
    for (const b of vis) { lo = Math.min(lo, b.l); hi = Math.max(hi, b.h); vMax = Math.max(vMax, b.v); }
    const range = hi - lo || 1;
    lo -= range * 0.05; hi += range * 0.05;
    const y = (p: number) => plotH - ((p - lo) / (hi - lo)) * plotH;
    const bw = plotW / vis.length;

    ctx.clearRect(0, 0, w, h);

    // grid + price labels
    ctx.font = "10px ui-monospace, Menlo, monospace";
    ctx.textBaseline = "middle";
    for (let i = 0; i <= 5; i++) {
      const p = lo + ((hi - lo) * i) / 5;
      const yy = y(p);
      ctx.strokeStyle = "rgba(255,255,255,0.05)";
      ctx.beginPath(); ctx.moveTo(0, yy); ctx.lineTo(plotW, yy); ctx.stroke();
      ctx.fillStyle = "#5a6478";
      ctx.fillText(p >= 100 ? p.toFixed(1) : p.toFixed(2), plotW + 8, yy);
    }
    // time labels
    ctx.textBaseline = "alphabetic";
    const step = Math.max(1, Math.floor(vis.length / 6));
    for (let i = 0; i < vis.length; i += step) {
      ctx.fillStyle = "#5a6478";
      ctx.fillText(vis[i].t.slice(0, 10), i * bw + 2, h - 8);
    }

    // volume
    for (let i = 0; i < vis.length; i++) {
      const b = vis[i];
      ctx.fillStyle = b.c >= b.o ? "rgba(52,211,153,0.25)" : "rgba(248,113,113,0.25)";
      const vh = vMax ? (b.v / vMax) * volH : 0;
      ctx.fillRect(i * bw + bw * 0.18, plotH + volH - vh, bw * 0.64, vh);
    }

    // candles
    for (let i = 0; i < vis.length; i++) {
      const b = vis[i];
      const up = b.c >= b.o;
      const color = up ? "#34d399" : "#f87171";
      const cx = i * bw + bw / 2;
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(1, bw * 0.08);
      ctx.beginPath(); ctx.moveTo(cx, y(b.h)); ctx.lineTo(cx, y(b.l)); ctx.stroke();
      ctx.fillStyle = color;
      const top = y(Math.max(b.o, b.c)), bot = y(Math.min(b.o, b.c));
      ctx.fillRect(i * bw + bw * 0.18, top, bw * 0.64, Math.max(1, bot - top));
    }

    // live last-price line
    const last = live ?? vis[vis.length - 1].c;
    if (last >= lo && last <= hi) {
      const yy = y(last);
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = "rgba(94,234,212,0.6)";
      ctx.beginPath(); ctx.moveTo(0, yy); ctx.lineTo(plotW, yy); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#0f2f2a";
      ctx.fillRect(plotW, yy - 9, padR, 18);
      ctx.fillStyle = "#5eead4";
      ctx.textBaseline = "middle";
      ctx.fillText(last >= 100 ? last.toFixed(1) : last.toFixed(2), plotW + 8, yy);
    }

    // crosshair + OHLCV readout
    if (cross && cross.x < plotW) {
      ctx.strokeStyle = "rgba(255,255,255,0.25)";
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(cross.x, 0); ctx.lineTo(cross.x, plotH + volH); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, cross.y); ctx.lineTo(plotW, cross.y); ctx.stroke();
      ctx.setLineDash([]);
      const i = Math.min(vis.length - 1, Math.max(0, Math.floor(cross.x / bw)));
      const b = vis[i];
      ctx.fillStyle = "rgba(6,10,20,0.85)";
      ctx.fillRect(8, 8, 330, 22);
      ctx.fillStyle = "#e8ecf4";
      ctx.textBaseline = "middle";
      ctx.fillText(
        `${b.t.slice(0, 10)}  O ${b.o.toFixed(2)}  H ${b.h.toFixed(2)}  L ${b.l.toFixed(2)}  C ${b.c.toFixed(2)}  V ${Intl.NumberFormat("en", { notation: "compact" }).format(b.v)}`,
        16, 19
      );
    }
  }, [bars, view, cross, live]);

  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    setView((v) => {
      const factor = e.deltaY > 0 ? 1.15 : 0.87;
      const count = Math.max(20, Math.min(bars.length, Math.round(v.count * factor)));
      const anchor = v.start + v.count; // keep right edge pinned
      return { count, start: Math.max(0, anchor - count) };
    });
  };
  const onMouseDown = (e: React.MouseEvent) => { drag.current = { x: e.clientX, start: view.start }; };
  const onMouseMove = (e: React.MouseEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    setCross({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    if (drag.current) {
      const dx = e.clientX - drag.current.x;
      const barsMoved = Math.round((dx / rect.width) * view.count);
      setView((v) => ({
        ...v,
        start: Math.max(0, Math.min(bars.length - v.count, drag.current!.start - barsMoved)),
      }));
    }
  };
  const stop = () => { drag.current = null; };

  return (
    <div className="chart-wrap" onMouseLeave={() => { setCross(null); stop(); }}>
      <canvas
        ref={canvasRef}
        onWheel={onWheel}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={stop}
      />
    </div>
  );
}
