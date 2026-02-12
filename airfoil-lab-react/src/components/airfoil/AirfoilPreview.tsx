'use client';

import { useMemo } from 'react';
import { genNaca4, nacaCodeFromMpt, rotateXY } from '@/lib/geometry';

interface AirfoilPreviewProps {
    camber: number;
    maxCamberPos: number;
    thickness: number;
    maxThicknessPos: number;
    alpha?: number;
    showGrid?: boolean;
    showMarkers?: boolean;
    cpData?: { x: number; cp: number; segment: 'upper' | 'lower' }[];
    showPressure?: boolean;
    className?: string;
}

export function AirfoilPreview({
    camber,
    maxCamberPos,
    thickness,
    maxThicknessPos,
    alpha = 0,
    showGrid = true,
    showMarkers = true,
    cpData = [],
    showPressure = false,
    className = '',
}: AirfoilPreviewProps) {
    const m = camber / 100;
    const p = maxCamberPos / 100;
    const t = thickness / 100;
    // Default to 30% if maxThicknessPos is missing or 0 (which is invalid for NACA 4-digit logic usually)
    const tpos = (maxThicknessPos || 30) / 100;

    const { x, y, nacaCode, upperCoords, lowerCoords } = useMemo(() => {
        const coords = genNaca4(m, p, t, tpos, 150);
        const code = nacaCodeFromMpt(m, p, t);

        // Split coords for upper/lower surface processing
        const mid = Math.floor(coords.x.length / 2);
        // Upper: trail -> lead (0..mid)
        // Lower: lead -> trail (mid..end)
        const ux = coords.x.slice(0, mid + 1);
        const uy = coords.y.slice(0, mid + 1);
        const lx = coords.x.slice(mid);
        const ly = coords.y.slice(mid);

        if (alpha !== 0) {
            const rotated = rotateXY(coords.x, coords.y, -alpha, 0.25, 0);
            const rUpper = rotateXY(ux, uy, -alpha, 0.25, 0);
            const rLower = rotateXY(lx, ly, -alpha, 0.25, 0);
            return {
                x: rotated.x, y: rotated.y, nacaCode: code,
                upperCoords: { x: rUpper.x, y: rUpper.y },
                lowerCoords: { x: rLower.x, y: rLower.y }
            };
        }
        return {
            x: coords.x, y: coords.y, nacaCode: code,
            upperCoords: { x: ux, y: uy },
            lowerCoords: { x: lx, y: ly }
        };
    }, [m, p, t, tpos, alpha]);

    const viewBox = "-0.2 -0.5 1.4 1.0"; // Expanded viewBox for pressure vectors
    const strokeWidth = 0.006;

    const pathD = useMemo(() => {
        if (x.length === 0) return '';
        let d = `M ${x[0]} ${-y[0]}`;
        for (let i = 1; i < x.length; i++) {
            d += ` L ${x[i]} ${-y[i]}`;
        }
        d += ' Z';
        return d;
    }, [x, y]);

    // Generate pressure vectors
    const pressureVectors = useMemo(() => {
        if (!showPressure || !cpData || cpData.length === 0) return [];

        return cpData.map((pt, i) => {
            // Find corresponding geometric point
            const coords = pt.segment === 'upper' ? upperCoords : lowerCoords;
            // Simple nearest neighbor search for coordinate (can be optimized)
            let idx = 0;
            let minD = 100;
            for (let j = 0; j < coords.x.length; j++) {
                const d = Math.abs(coords.x[j] - pt.x);
                if (d < minD) { minD = d; idx = j; }
            }

            const px = coords.x[idx];
            const py = -coords.y[idx]; // SVG Y is inverted

            // Calculate normal vector (approximate using neighbors)
            const pPrev = idx > 0 ? { x: coords.x[idx - 1], y: -coords.y[idx - 1] } : { x: px, y: py };
            const pNext = idx < coords.x.length - 1 ? { x: coords.x[idx + 1], y: -coords.y[idx + 1] } : { x: px, y: py };
            const dx = pNext.x - pPrev.x;
            const dy = pNext.y - pPrev.y;

            // Normal vector (-dy, dx) normalized
            const len = Math.sqrt(dx * dx + dy * dy);
            let nx = -dy / (len || 1);
            let ny = dx / (len || 1);

            // Ensure normal points OUTWARD
            // Upper surface (y > 0 usually, SVG y < 0): Normal should point UP (neg SVG y)
            // Lower surface (y < 0 usually, SVG y > 0): Normal should point DOWN (pos SVG y)
            if (pt.segment === 'upper' && ny > 0) { nx = -nx; ny = -ny; }
            if (pt.segment === 'lower' && ny < 0) { nx = -nx; ny = -ny; }

            const scale = 0.15; // Visual scale factor
            const cpVal = pt.cp;
            const vecLen = Math.abs(cpVal) * scale;

            // Cp negative (Suction) -> Arrow points OUT (away from surface)
            // Cp positive (Pressure) -> Arrow points IN (towards surface)
            const isSuction = cpVal < 0;

            const startX = px;
            const startY = py;
            let endX, endY;

            if (isSuction) {
                // Point OUT: start at surface, end away
                endX = startX + nx * vecLen;
                endY = startY + ny * vecLen;
            } else {
                // Point IN: start away, end at surface
                endX = startX + nx * vecLen;
                endY = startY + ny * vecLen;
                // Actually for "pushing", we typically draw arrow pointing TO surface.
                // So we can draw from (startX + nx*vecLen) TO (startX). 
                // But let's keep visual consistency:
                // Suction (pull): Surface -> Out
                // Pressure (push): Out -> Surface
            }

            return {
                x1: isSuction ? startX : startX + nx * vecLen,
                y1: isSuction ? startY : startY + ny * vecLen,
                x2: isSuction ? startX + nx * vecLen : startX,
                y2: isSuction ? startY + ny * vecLen : startY,
                color: isSuction ? '#60a5fa' : '#f87171', // Blue (suction), Red (pressure)
                cp: cpVal
            };
        });
    }, [cpData, showPressure, upperCoords, lowerCoords]);

    // Helper to get color for Cp value
    const getCpColor = (cp: number) => {
        // -Cp is suction (blue/cyan), +Cp is pressure (red/orange)
        // Range roughly -2 to +1
        if (cp < 0) {
            // Suction: interpolate -2 (dark blue) to 0 (white/light-blue)
            const t = Math.min(1, Math.abs(cp) / 2);
            // blue-500: #3b82f6 -> cyan-300: #67e8f9
            return `rgba(${59 + (103 - 59) * t}, ${130 + (232 - 130) * t}, ${246 + (249 - 246) * t}, 1)`;
        } else {
            // Pressure: 0 (white) to 1 (red)
            const t = Math.min(1, cp / 1);
            // red-500: #ef4444
            return `rgba(239, 68, 68, ${0.5 + 0.5 * t})`;
        }
    };

    // Generate heatmap segments
    const heatmapSegments = useMemo(() => {
        if (!showPressure || !cpData || cpData.length === 0) return null;

        const segments: React.ReactNode[] = [];

        // Helper to process surface
        const processSurface = (coords: { x: number[], y: number[] }, surfaceType: 'upper' | 'lower') => {
            for (let i = 0; i < coords.x.length - 1; i++) {
                const x1 = coords.x[i];
                const y1 = -coords.y[i];
                const x2 = coords.x[i + 1];
                const y2 = -coords.y[i + 1];

                // Find Cp for this segment (average of ends or nearest)
                // We'll use the midpoint's x to find nearest Cp
                const midX = (x1 + x2) / 2;
                const closestPt = cpData.reduce((prev, curr) =>
                    (curr.segment === surfaceType && Math.abs(curr.x - midX) < Math.abs(prev.x - midX)) ? curr : prev
                );

                const color = closestPt.cp < 0 ? '#3b82f6' : '#ef4444'; // Simple Blue/Red for now
                const opacity = Math.min(1, Math.abs(closestPt.cp) / 1.5 + 0.2); // Opacity based on magnitude

                segments.push(
                    <line
                        key={`${surfaceType}-${i}`}
                        x1={x1} y1={y1} x2={x2} y2={y2}
                        stroke={color}
                        strokeWidth={0.015} // Thicker for visibility
                        strokeOpacity={opacity}
                        strokeLinecap="round"
                    />
                );
            }
        };

        processSurface(upperCoords, 'upper');
        processSurface(lowerCoords, 'lower');

        return segments;
    }, [cpData, showPressure, upperCoords, lowerCoords]);

    return (
        <div className={`relative ${className}`}>
            <svg
                viewBox={viewBox}
                className="w-full h-full"
                style={{ backgroundColor: 'transparent' }}
            >
                {/* Grid lines - subtle dark theme */}
                {showGrid && (
                    <g stroke="rgba(148, 163, 184, 0.08)" strokeWidth={0.002}>
                        {[0, 0.25, 0.5, 0.75, 1].map((xi) => (
                            <line key={`v-${xi}`} x1={xi} y1={-0.35} x2={xi} y2={0.35} />
                        ))}
                        {[-0.2, -0.1, 0, 0.1, 0.2].map((yi) => (
                            <line key={`h-${yi}`} x1={-0.1} y1={yi} x2={1.1} y2={yi} />
                        ))}
                    </g>
                )}

                {/* Chord line */}
                <line
                    x1={0} y1={0} x2={1} y2={0}
                    stroke="rgba(148, 163, 184, 0.2)"
                    strokeWidth={0.003}
                    strokeDasharray="0.02 0.01"
                />

                {/* Pressure Vectors (Optional - can be toggleable separately or overlay) */}
                {showPressure && pressureVectors.map((vec, i) => (
                    <line
                        key={`vec-${i}`}
                        x1={vec.x1} y1={vec.y1} x2={vec.x2} y2={vec.y2}
                        stroke={vec.color}
                        strokeWidth={0.003}
                        opacity={0.6}
                    />
                ))}

                {/* Airfoil glow effect (reduced if heatmap is on) */}
                {!showPressure && (
                    <path
                        d={pathD}
                        fill="none"
                        stroke="rgba(96, 165, 250, 0.15)"
                        strokeWidth={0.025}
                        strokeLinejoin="round"
                        filter="url(#glow)"
                    />
                )}

                {/* Airfoil fill + stroke (Base) */}
                <path
                    d={pathD}
                    fill="url(#airfoilGradientDark)"
                    stroke={(showPressure && cpData && cpData.length > 0) ? "none" : "#60a5fa"} // Only hide stroke if heatmap is actually rendered
                    strokeWidth={strokeWidth}
                    strokeLinejoin="round"
                />

                {/* Heatmap Overlay */}
                {showPressure && heatmapSegments}

                {/* Markers */}
                {showMarkers && (
                    <g>
                        <line
                            x1={p} y1={-0.25} x2={p} y2={0.25}
                            stroke="#34d399"
                            strokeWidth={0.003}
                            strokeDasharray="0.015 0.008"
                            opacity={0.5}
                        />
                        <line
                            x1={tpos} y1={-0.25} x2={tpos} y2={0.25}
                            stroke="#fb923c"
                            strokeWidth={0.003}
                            strokeDasharray="0.01 0.005"
                            opacity={0.5}
                        />
                    </g>
                )}

                {/* Definitions */}
                <defs>
                    <linearGradient id="airfoilGradientDark" x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stopColor="#60a5fa" stopOpacity={0.2} />
                        <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.05} />
                    </linearGradient>
                    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                        <feGaussianBlur stdDeviation="0.01" result="blur" />
                    </filter>
                </defs>
            </svg>

            {/* Info label */}
            <div className="absolute bottom-2 left-3 text-xs text-slate-500 font-mono">
                NACA {nacaCode} · t<sub>pos</sub>={maxThicknessPos.toFixed(0)}%
                {alpha !== 0 && ` · α=${alpha.toFixed(1)}°`}
            </div>

            {/* Legend */}
            {showMarkers && !showPressure && (
                <div className="absolute top-2 right-3 text-[10px] space-y-1">
                    <div className="flex items-center gap-1.5">
                        <span className="w-3 h-0.5 bg-emerald-400 rounded-full"></span>
                        <span className="text-slate-500">Max camber</span>
                    </div>
                </div>
            )}
        </div>
    );
}

export default AirfoilPreview;
