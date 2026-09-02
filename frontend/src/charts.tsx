/* SVG charts.
   Hand-rolled rather than pulled from a library: these are four specific
   forms, they need to inherit the dashboard's theme tokens, and the whole app
   stays dependency-free.

   Palette: categorical slots 1-3 of the validated default palette
   (blue / orange / aqua), stepped separately for light and dark surfaces and
   checked with the palette validator against this dashboard's own surfaces
   (all-pairs CVD ΔE 9.2 light / 9.4 dark, normal-vision 24.0 / 20.9).
   Aqua falls below 3:1 on the light surface, so every chart that uses it also
   carries a legend and direct labels — identity is never colour alone. */

import type { ReactNode } from 'react';
import { useState } from 'react';

export const SERIES = ['var(--series-1)', 'var(--series-2)', 'var(--series-3)'];

const PAD = { top: 12, right: 16, bottom: 26, left: 46 };

interface Point {
  x: number;
  y: number;
  label?: string;
}

function useTooltip() {
  const [tip, setTip] = useState<{ x: number; y: number; content: ReactNode } | null>(null);
  return { tip, setTip };
}

function Axes({
  width,
  height,
  yTicks,
  xTicks,
  yFormat,
}: {
  width: number;
  height: number;
  yTicks: number[];
  xTicks: { value: number; label: string }[];
  yFormat: (value: number) => string;
}) {
  return (
    <g>
      {yTicks.map((tick, index) => (
        <g key={index}>
          <line
            x1={PAD.left}
            x2={width - PAD.right}
            y1={tick}
            y2={tick}
            stroke="var(--border)"
            strokeWidth={1}
          />
        </g>
      ))}
      {yTicks.map((tick, index) => (
        <text
          key={`l${index}`}
          x={PAD.left - 8}
          y={tick + 3.5}
          textAnchor="end"
          fontSize={10}
          fill="var(--text-faint)"
          fontFamily="var(--font-mono)"
        >
          {yFormat(index)}
        </text>
      ))}
      {xTicks.map((tick, index) => (
        <text
          key={`x${index}`}
          x={tick.value}
          y={height - PAD.bottom + 15}
          textAnchor="middle"
          fontSize={10}
          fill="var(--text-faint)"
          fontFamily="var(--font-mono)"
        >
          {tick.label}
        </text>
      ))}
    </g>
  );
}

/* ------------------------------------------------------------ line chart */

export function LineChart({
  series,
  height = 200,
  yLabel,
  valueFormat = (v: number) => v.toFixed(2),
  zeroLine = false,
}: {
  series: { name: string; points: Point[] }[];
  height?: number;
  yLabel?: string;
  valueFormat?: (value: number) => string;
  zeroLine?: boolean;
}) {
  const width = 720;
  const { tip, setTip } = useTooltip();
  const all = series.flatMap((s) => s.points);
  if (!all.length) return <div className="empty">No data to plot</div>;

  const xs = all.map((p) => p.x);
  const ys = all.map((p) => p.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  let yMin = Math.min(...ys);
  let yMax = Math.max(...ys);
  if (zeroLine) {
    yMin = Math.min(yMin, 0);
    yMax = Math.max(yMax, 0);
  }
  const yPad = (yMax - yMin) * 0.08 || Math.abs(yMax || 1) * 0.1;
  yMin -= yPad;
  yMax += yPad;

  const sx = (x: number) =>
    PAD.left + ((x - xMin) / (xMax - xMin || 1)) * (width - PAD.left - PAD.right);
  const sy = (y: number) =>
    height - PAD.bottom - ((y - yMin) / (yMax - yMin || 1)) * (height - PAD.top - PAD.bottom);

  const gridCount = 4;
  const yTickValues = Array.from(
    { length: gridCount + 1 },
    (_, i) => yMin + ((yMax - yMin) * i) / gridCount,
  );
  const xTickValues = [0, 0.25, 0.5, 0.75, 1].map((f) => xMin + (xMax - xMin) * f);

  return (
    <div style={{ position: 'relative' }}>
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height} role="img"
           aria-label={yLabel ?? 'line chart'}>
        <Axes
          width={width}
          height={height}
          yTicks={yTickValues.map(sy)}
          xTicks={xTickValues.map((value) => ({
            value: sx(value),
            label: series[0]?.points.find((p) => p.x >= value)?.label ?? '',
          }))}
          yFormat={(index) => valueFormat(yTickValues[index])}
        />
        {zeroLine && yMin < 0 && yMax > 0 && (
          <line
            x1={PAD.left}
            x2={width - PAD.right}
            y1={sy(0)}
            y2={sy(0)}
            stroke="var(--border-strong)"
            strokeWidth={1}
            strokeDasharray="3 3"
          />
        )}
        {series.map((s, index) => {
          const path = s.points
            .map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`)
            .join(' ');
          return (
            <g key={s.name}>
              <path
                d={path}
                fill="none"
                stroke={SERIES[index % SERIES.length]}
                strokeWidth={2}
                strokeLinejoin="round"
                strokeLinecap="round"
              />
              {s.points.length <= 40 &&
                s.points.map((p, i) => (
                  <circle
                    key={i}
                    cx={sx(p.x)}
                    cy={sy(p.y)}
                    r={4}
                    fill={SERIES[index % SERIES.length]}
                    stroke="var(--bg-raised)"
                    strokeWidth={2}
                    onMouseEnter={() =>
                      setTip({
                        x: sx(p.x),
                        y: sy(p.y),
                        content: `${s.name} · ${p.label ?? ''} ${valueFormat(p.y)}`,
                      })
                    }
                    onMouseLeave={() => setTip(null)}
                  />
                ))}
              {/* Direct end label: identity without relying on colour alone. */}
              {s.points.length > 0 && (
                <text
                  x={sx(s.points[s.points.length - 1].x) - 4}
                  y={sy(s.points[s.points.length - 1].y) - 8}
                  textAnchor="end"
                  fontSize={10}
                  fontWeight={600}
                  fill="var(--text-secondary, var(--text-muted))"
                  fontFamily="var(--font-mono)"
                >
                  {s.name}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      {tip && (
        <div
          className="chart-tip"
          style={{ left: `${(tip.x / width) * 100}%`, top: tip.y - 34 }}
        >
          {tip.content}
        </div>
      )}
    </div>
  );
}

/* ----------------------------------------------------- calibration chart */

export function CalibrationChart({
  points,
  height = 260,
}: {
  points: { predicted: number; observed: number; count: number; bin?: string }[];
  height?: number;
}) {
  const width = 420;
  const { tip, setTip } = useTooltip();
  if (!points.length) return <div className="empty">No calibration data yet</div>;

  const plotW = width - PAD.left - PAD.right;
  const plotH = height - PAD.top - PAD.bottom;
  const sx = (v: number) => PAD.left + v * plotW;
  const sy = (v: number) => height - PAD.bottom - v * plotH;
  const maxCount = Math.max(...points.map((p) => p.count));

  return (
    <div style={{ position: 'relative' }}>
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height} role="img"
           aria-label="calibration curve">
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line x1={sx(0)} x2={sx(1)} y1={sy(tick)} y2={sy(tick)} stroke="var(--border)" />
            <text x={PAD.left - 8} y={sy(tick) + 3.5} textAnchor="end" fontSize={10}
                  fill="var(--text-faint)" fontFamily="var(--font-mono)">
              {(tick * 100).toFixed(0)}
            </text>
            <text x={sx(tick)} y={height - PAD.bottom + 15} textAnchor="middle" fontSize={10}
                  fill="var(--text-faint)" fontFamily="var(--font-mono)">
              {(tick * 100).toFixed(0)}
            </text>
          </g>
        ))}
        {/* Reference: perfect calibration. A guide, not a data series. */}
        <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(1)} stroke="var(--border-strong)"
              strokeWidth={1.5} strokeDasharray="4 4" />
        <text x={sx(0.72)} y={sy(0.78)} fontSize={10} fill="var(--text-faint)"
              fontFamily="var(--font-mono)">
          perfect
        </text>
        {points.map((point, index) => (
          <circle
            key={index}
            cx={sx(point.predicted)}
            cy={sy(point.observed)}
            r={5 + 7 * Math.sqrt(point.count / (maxCount || 1))}
            fill={SERIES[0]}
            fillOpacity={0.75}
            stroke="var(--bg-raised)"
            strokeWidth={2}
            onMouseEnter={() =>
              setTip({
                x: sx(point.predicted),
                y: sy(point.observed),
                content: `predicted ${(point.predicted * 100).toFixed(1)}% · actual ${(point.observed * 100).toFixed(1)}% · n=${point.count}`,
              })
            }
            onMouseLeave={() => setTip(null)}
          />
        ))}
        <text x={sx(0.5)} y={height - 2} textAnchor="middle" fontSize={10}
              fill="var(--text-muted)">
          predicted %
        </text>
      </svg>
      {tip && (
        <div className="chart-tip" style={{ left: `${(tip.x / width) * 100}%`, top: tip.y - 36 }}>
          {tip.content}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------ bucket bars */

export function BucketBars({
  data,
  height = 200,
  valueFormat = (v: number) => `${(v * 100).toFixed(1)}%`,
  label = 'value',
}: {
  data: { bucket: string; value: number; count?: number }[];
  height?: number;
  valueFormat?: (value: number) => string;
  label?: string;
}) {
  const width = 560;
  const { tip, setTip } = useTooltip();
  if (!data.length) return <div className="empty">No data yet</div>;

  const values = data.map((d) => d.value);
  const max = Math.max(...values, 0);
  const min = Math.min(...values, 0);
  const span = max - min || 1;
  const plotH = height - PAD.top - PAD.bottom;
  const zeroY = PAD.top + (max / span) * plotH;
  const slot = (width - PAD.left - PAD.right) / data.length;
  const barWidth = Math.min(slot - 10, 62);

  return (
    <div style={{ position: 'relative' }}>
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height} role="img"
           aria-label={`${label} by bucket`}>
        <line x1={PAD.left} x2={width - PAD.right} y1={zeroY} y2={zeroY}
              stroke="var(--border-strong)" strokeWidth={1} />
        {data.map((item, index) => {
          const x = PAD.left + slot * index + (slot - barWidth) / 2;
          const barHeight = (Math.abs(item.value) / span) * plotH;
          const y = item.value >= 0 ? zeroY - barHeight : zeroY;
          // Polarity, so a diverging pair: blue above zero, orange below.
          const fill = item.value >= 0 ? SERIES[0] : SERIES[1];
          return (
            <g key={item.bucket}
               onMouseEnter={() =>
                 setTip({
                   x: x + barWidth / 2,
                   y: Math.min(y, zeroY),
                   content: `${item.bucket}: ${valueFormat(item.value)}${item.count ? ` · n=${item.count}` : ''}`,
                 })
               }
               onMouseLeave={() => setTip(null)}>
              <rect x={x} y={y} width={barWidth} height={Math.max(barHeight, 1)}
                    fill={fill} rx={4} />
              <text x={x + barWidth / 2}
                    y={item.value >= 0 ? y - 5 : y + barHeight + 12}
                    textAnchor="middle" fontSize={10} fontWeight={600}
                    fill="var(--text-muted)" fontFamily="var(--font-mono)">
                {valueFormat(item.value)}
              </text>
              <text x={x + barWidth / 2} y={height - 8} textAnchor="middle" fontSize={10}
                    fill="var(--text-faint)">
                {item.bucket}
              </text>
            </g>
          );
        })}
      </svg>
      {tip && (
        <div className="chart-tip" style={{ left: `${(tip.x / width) * 100}%`, top: tip.y - 34 }}>
          {tip.content}
        </div>
      )}
    </div>
  );
}

export function Legend({ items }: { items: { name: string; index: number }[] }) {
  return (
    <div className="row" style={{ gap: 14, fontSize: '0.75rem' }}>
      {items.map((item) => (
        <span key={item.name} className="row" style={{ gap: 5 }}>
          <span
            className="dot"
            style={{ background: SERIES[item.index % SERIES.length] }}
          />
          <span className="muted">{item.name}</span>
        </span>
      ))}
    </div>
  );
}
