'use client';

import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    ReferenceLine,
} from 'recharts';
import { CpDataPoint } from '@/types';

interface CpChartProps {
    data: CpDataPoint[];
    nacaCode: string;
    alpha: number;
    className?: string;
}

const darkAxisStyle = { fontSize: 11, fill: '#94a3b8' };
const darkLabelStyle = { fill: '#94a3b8', fontSize: 12 };

export function CpChart({ data, nacaCode, alpha, className = '' }: CpChartProps) {
    // Separate upper and lower surface data, sorted by x
    const upperRaw = data
        .filter(d => d.segment === 'upper')
        .sort((a, b) => a.x - b.x);
    const lowerRaw = data
        .filter(d => d.segment === 'lower')
        .sort((a, b) => a.x - b.x);

    // Merge into a single dataset keyed by x
    const xSet = new Set<number>();
    upperRaw.forEach(d => xSet.add(Math.round(d.x * 1000) / 1000));
    lowerRaw.forEach(d => xSet.add(Math.round(d.x * 1000) / 1000));
    const xValues = Array.from(xSet).sort((a, b) => a - b);

    const chartData = xValues.map(x => {
        const u = upperRaw.find(d => Math.abs(d.x - x) < 0.002);
        const l = lowerRaw.find(d => Math.abs(d.x - x) < 0.002);
        return {
            x: parseFloat(x.toFixed(3)),
            upper: u ? parseFloat(u.cp.toFixed(4)) : undefined,
            lower: l ? parseFloat(l.cp.toFixed(4)) : undefined,
        };
    });

    if (chartData.length === 0) {
        return (
            <div className={`flex items-center justify-center h-[300px] text-slate-500 text-sm ${className}`}>
                Run simulation to see Cp distribution
            </div>
        );
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const CustomTooltip = ({ active, payload, label }: any) => {
        if (!active || !payload?.length) return null;
        return (
            <div className="bg-[hsl(220,20%,14%)] border border-[hsl(220,15%,25%)] rounded-lg px-3 py-2 shadow-xl text-xs">
                <p className="text-slate-400 mb-1">x/c = {typeof label === 'number' ? label.toFixed(3) : label}</p>
                {payload.map((entry: { name: string; value: number; color: string }, i: number) => (
                    <p key={i} style={{ color: entry.color }}>
                        {entry.name === 'upper' ? 'Upper' : 'Lower'}: Cp = {entry.value?.toFixed(4)}
                    </p>
                ))}
            </div>
        );
    };

    return (
        <div className={className}>
            <p className="text-xs text-slate-500 mb-2">
                NACA {nacaCode} · α = {alpha.toFixed(1)}° · <span className="text-blue-400">■</span> Upper · <span className="text-red-400">■</span> Lower
            </p>
            <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                    <XAxis
                        dataKey="x"
                        type="number"
                        domain={[0, 1]}
                        tick={darkAxisStyle}
                        label={{ value: 'x/c', position: 'insideBottomRight', offset: -5, ...darkLabelStyle }}
                    />
                    <YAxis
                        reversed
                        tick={darkAxisStyle}
                        label={{ value: 'Cp', angle: -90, position: 'insideLeft', offset: 10, ...darkLabelStyle }}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <ReferenceLine y={0} stroke="rgba(148,163,184,0.3)" strokeDasharray="4 4" />
                    <Line
                        type="monotone"
                        dataKey="upper"
                        stroke="#60a5fa"
                        strokeWidth={2}
                        dot={false}
                        name="upper"
                        connectNulls
                    />
                    <Line
                        type="monotone"
                        dataKey="lower"
                        stroke="#f87171"
                        strokeWidth={2}
                        dot={false}
                        name="lower"
                        connectNulls
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}

export default CpChart;
